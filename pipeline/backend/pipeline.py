#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import copy
import pickle
import pprint
from types import SimpleNamespace

from pipeline.backend.config import Backend, WorkMode
from pipeline.backend.config import Role
from pipeline.backend.config import StatusCode
from pipeline.backend.config import VERSION
from pipeline.backend.task_info import TaskInfo
from pipeline.component.component_base import Component
from pipeline.component.input import Input
from pipeline.interface.data import Data
from pipeline.interface.model import Model
from pipeline.utils import tools
from pipeline.utils.invoker.job_submitter import JobInvoker


class PipeLine(object):
    def __init__(self):
        self._initiator = None
        self._roles = {}
        self._components = {}
        self._components_input = {}
        self._train_dsl = {}
        self._predict_dsl = {}
        self._train_conf = {}
        self._upload_conf = []
        self._cur_state = None
        self._job_invoker = JobInvoker()
        self._train_job_id = None
        self._predict_job_id = None
        self._fit_status = None
        self._train_board_url = None
        self._model_info = None
        # self._dsl_parser = DSLParser()

    def set_initiator(self, role, party_id):
        self._initiator = SimpleNamespace(role=role, party_id=party_id)

        return self

    def _get_initiator_conf(self):
        if self._initiator is None:
            raise ValueError("Please set initiator of PipeLine")

        initiator_conf = {"role": self._initiator.role,
                          "party_id": self._initiator.party_id}

        return initiator_conf

    def set_roles(self, guest=None, host=None, arbiter=None, **kwargs):
        local_parameters = locals()
        support_roles = Role.support_roles()
        for role, party_id in local_parameters.items():
            if role == "self":
                continue

            if not local_parameters.get(role):
                continue

            if role not in support_roles:
                raise ValueError("Current role not support {}, support role list {}".format(role, support_roles))

            party_id = local_parameters.get(role)
            self._roles[role] = []
            if isinstance(party_id, int):
                self._roles[role].append(party_id)
            elif isinstance(party_id, list):
                self._roles[role].extend(party_id)
            else:
                raise ValueError("role: {}'s party_id should be an integer or a list of integer".format(role))

        return self

    def _get_role_conf(self):
        return self._roles

    def _get_party_index(self, role, party_id):
        if role not in self._roles:
            raise ValueError("role {} does not setting".format(role))

        if party_id not in self._roles[role]:
            raise ValueError("role {} does not init setting with the party_id {}".format(role, party_id))

        return self._roles[role].index(party_id)

    def set_deploy_end_component(self, components):
        if not isinstance(components, list):
            components = [components]

        for idx in range(len(components)):
            if isinstance(components[idx], str):
                if components[idx] not in self._components:
                    raise ValueError("component name {} does not exist".format(components[idx]))
            elif isinstance(components[idx], Component):
                component_name = components[idx].name
                if component_name not in self._components:
                    raise ValueError("component name {} does not exist".format(component_name))

                components[idx] = component_name
            else:
                raise ValueError("input parameter should be component name or component object")

        return self

    def add_component(self, component, data=None, model=None):
        if not isinstance(component, Component):
            raise ValueError(
                "To add a component to pipeline, component {} should be a Component object".format(component))

        if component.name in self._components:
            raise Warning("component {} is added before".format(component.name))

        self._components[component.name] = component

        if data is not None:
            if not isinstance(data, Data):
                raise ValueError("data input of component {} should be passed by data object".format(component.name))

            attrs_dict = vars(data)
            self._components_input[component.name] = {"data": {}}
            for attr, val in attrs_dict.items():
                if not attr.endswith("data"):
                    continue

                if val is None:
                    continue

                print("data dep ", attr, val)
                print("add dep ", component.name, attr)
                data_key = attr.strip("_")
                # if data_key == "validate_data" or data_key == "test_data":
                #     data_key = "eval_data"

                if isinstance(val, list):
                    self._components_input[component.name]["data"][data_key] = val
                else:
                    self._components_input[component.name]["data"][data_key] = [val]

        if model is not None:
            if not isinstance(model, Model):
                raise ValueError("model input of component {} should be passed by model object".format(component.name))

            attrs_dict = vars(model)
            for attr, val in attrs_dict.items():
                if not attr.endswith("model"):
                    continue

                if val is None:
                    continue

                if isinstance(val, list):
                    self._components_input[component.name][attr.strip("_")] = val
                else:
                    self._components_input[component.name][attr.strip("_")] = [val]

    def add_upload_data(self, file, table_name, namespace, head=1, partition=16):
        data_conf = {"file": file,
                     "table_name": table_name,
                     "namespace": namespace,
                     "head": head,
                     "partition": partition}
        self._upload_conf.append(data_conf)

    def _get_task_inst(self, job_id, name, init_role, party_id):
        return TaskInfo(jobid=job_id,
                        component=self._components[name],
                        job_client=self._job_invoker,
                        role=init_role,
                        party_id=party_id)

    def get_component(self, component_names=None):
        job_id = self._train_job_id
        if self._cur_state != "fit":
            job_id = self._predict_job_id

        init_role = self._initiator.role
        party_id = self._initiator.party_id
        if not component_names:
            component_tasks = {}
            for name in self._components:
                component_tasks[name] = self._get_task_inst(job_id, name, init_role, party_id)
            return component_tasks
        elif isinstance(component_names, str):
            return self._get_task_inst(job_id, component_names, init_role, party_id)
        elif isinstance(component_names, list):
            component_tasks = []
            for name in component_names:
                component_tasks.append(self._get_task_inst(job_id, name, init_role, party_id))

            return component_tasks

    def _construct_train_dsl(self):
        self._train_dsl["components"] = {}
        for name, component in self._components.items():
            component_dsl = {"module": component.module}
            if name in self._components_input:
                component_dsl["input"] = self._components_input[name]

            if hasattr(component, "output"):
                component_dsl["output"] = {}
                if hasattr(component.output, "data_output"):
                    component_dsl["output"]["data"] = component.output.data_output

                if hasattr(component.output, "model"):
                    component_dsl["output"]["model"] = component.output.model_output

            self._train_dsl["components"][name] = component_dsl

        if not self._train_dsl:
            raise ValueError("there are no components to train")

        print("train_dsl : ", self._train_dsl)

    def _construct_train_conf(self):
        self._train_conf["initiator"] = self._get_initiator_conf()
        self._train_conf["role"] = self._roles
        self._train_conf["job_parameters"] = self._get_job_parameters(job_type="train", version=2)
        self._train_conf["role_parameters"] = {}
        for name, component in self._components.items():
            param_conf = component.get_config(version=VERSION, roles=self._roles)

            if "algorithm_parameters" in param_conf:
                algorithm_param_conf = param_conf["algorithm_parameters"]
                if "algorithm_parameters" not in self._train_conf:
                    self._train_conf["algorithm_parameters"] = {}
                self._train_conf["algorithm_parameters"].update(algorithm_param_conf)

            if "role_parameters" in param_conf:
                role_param_conf = param_conf["role_parameters"]
                self._train_conf["role_parameters"] = tools.merge_dict(role_param_conf,
                                                                       self._train_conf["role_parameters"])

        import pprint
        pprint.pprint(self._train_conf)
        return self._train_conf

    def _get_job_parameters(self, job_type="train", backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE, version=2):
        job_parameters = {
            "job_type": job_type,
            "backend": backend.value,
            "work_mode": work_mode.value,
            "dsl_version": version
        }

        return job_parameters

    def _construct_upload_conf(self, data_conf, work_mode):
        upload_conf = copy.deepcopy(data_conf)
        upload_conf["work_mode"] = work_mode
        return upload_conf

    def get_train_job_id(self):
        return self._train_job_id

    def get_predict_job_id(self):
        return self._predict_job_id

    def _set_state(self, state):
        self._cur_state = state

    def compile(self):
        self._construct_train_dsl()
        self._train_conf = self._construct_train_conf()

    def _feed_data_and_job_parameters(self, conf, feed_dict, backend, work_mode, job_type=None, model_info=None):
        submit_conf = copy.deepcopy(conf)
        print("submit conf' type {}".format(submit_conf))
        submit_conf["job_parameters"] = {
            "work_mode": work_mode.value,
            "backend": backend.value,
            "dsl_version": VERSION
        }

        if job_type is not None:
            submit_conf["job_parameters"]["job_type"] = job_type

        if model_info is not None:
            submit_conf["job_parameters"]["model_id"] = model_info.model_id
            submit_conf["job_parameters"]["model_version"] = model_info.model_version

        if not isinstance(feed_dict, dict):
            return submit_conf

        data_source = {}
        for _input, _input_dict in feed_dict.items():
            print("\n\n\n", _input, _input_dict)
            if not isinstance(_input, Input):
                raise ValueError("key of feed_dict should an input object of the name of an input object")

            for role in _input_dict:
                if role not in data_source:
                    data_source[role] = {}

                if VERSION == 1:
                    if "args" not in data_source[role]:
                        data_source[role]["args"] = {"data": {}}

                    all_party = self._get_role_conf()[role]
                    data = [{}] * len(all_party)
                    for idx in range(len(all_party)):
                        _party_id = all_party[idx]
                        if _party_id not in _input_dict[role]:
                            raise ValueError(
                                "In Pipeline, to use dsl version-1, all party's data in role should be list, but role: {}, party: {} not found".format(
                                    role, _party_id))
                        data[idx] = _input_dict[role][_party_id]

                    if job_type != "predict":
                        data_source[role]["args"]["data"][_input.name] = data
                    else:
                        data_source[role]["args"]["data"]["eval_data"] = data

                    continue

                for _party_id in _input_dict[role]:
                    _party_index = str(self._get_party_index(role, _party_id))
                    if _party_index not in data_source[role]:
                        data_source[role][_party_index] = {}
                        data_source[role][_party_index]["args"] = {}

                    data_source[role][_party_index]["args"][_input.name] = _input_dict[role][_party_id]

        if "role_parameters" not in submit_conf:
            submit_conf["role_parameters"] = data_source
        else:
            all_roles = set(submit_conf["role_parameters"].keys()) | set(data_source.keys())
            for role in all_roles:
                if role not in submit_conf["role_parameters"]:
                    submit_conf["role_parameters"][role] = data_source[role]
                elif role in data_source:
                    submit_conf["role_parameters"][role] = tools.merge_dict(data_source[role],
                                                                            submit_conf["role_parameters"][role])

        import pprint
        pprint.pprint(submit_conf)
        return submit_conf

    def fit(self, backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE, feed_dict=None):
        print("_train_conf {}".format(self._train_conf))
        self._set_state("fit")
        training_conf = self._feed_data_and_job_parameters(self._train_conf, feed_dict, backend, work_mode)
        pprint.pprint(training_conf)
        self._train_job_id, detail_info = self._job_invoker.submit_job(self._train_dsl, training_conf)
        self._train_board_url = detail_info["board_url"]
        self._model_info = SimpleNamespace(model_id=detail_info["model_info"]["model_id"],
                                           model_version=detail_info["model_info"]["model_version"])

        self._fit_status = self._job_invoker.monitor_job_status(self._train_job_id,
                                                                self._initiator.role,
                                                                self._initiator.party_id)

    def predict(self, backend=Backend.EGGROLL, work_mode=WorkMode.CLUSTER, feed_dict=None):
        if self._fit_status != StatusCode.SUCCESS:
            print("Pipeline should be fit successfully before predict!!!")
            return

        predict_conf = self._feed_data_and_job_parameters(self._train_conf,
                                                          feed_dict,
                                                          backend,
                                                          work_mode,
                                                          job_type="predict",
                                                          model_info=self._model_info)

        self._predict_job_id, _ = self._job_invoker.submit_job(dsl=self._predict_dsl, submit_conf=predict_conf)
        self._job_invoker.monitor_job_status(self._predict_job_id,
                                             self._initiator.role,
                                             self._initiator.party_id)

    def upload(self, work_mode=WorkMode.STANDALONE, drop=0):
        for data_conf in self._upload_conf:
            upload_conf = self._construct_upload_conf(data_conf, work_mode)
            self._train_job_id, detail_info = self._job_invoker.upload_data(upload_conf, int(drop))
            self._train_board_url = detail_info["board_url"]
            self._job_invoker.monitor_job_status(self._train_job_id,
                                                 "local",
                                                 0)

    def dump(self):
        return pickle.dumps(self)

    @classmethod
    def load(cls, pipeline_bytes):
        return pickle.loads(pipeline_bytes)

    def deploy_component(self, components):
        if self._train_dsl is None:
            raise ValueError("before deploy model, training should be finish!!!")

        if not components:
            deploy_cpns = list(self._components.keys())
        else:
            deploy_cpns = []
            for cpn in components:
                if isinstance(cpn, str):
                    deploy_cpns.append(cpn)
                elif isinstance(cpn, Component):
                    deploy_cpns.append(cpn.name)
                else:
                    raise ValueError(
                        "deploy component parameters is wrong, expect str or Component object, but {} find".format(cpn))

        self._predict_dsl = self._job_invoker.get_predict_dsl(train_dsl=self._train_dsl, cpn_list=deploy_cpns,
                                                              version=VERSION)

        return self
