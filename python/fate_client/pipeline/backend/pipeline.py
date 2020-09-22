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
import json
import pickle
import sys
import time
from types import SimpleNamespace

from pipeline.backend.config import Backend, WorkMode
from pipeline.backend.config import Role
from pipeline.backend.config import StatusCode
from pipeline.backend.config import VERSION
from pipeline.backend.task_info import TaskInfo
from pipeline.component.component_base import Component
from pipeline.component.reader import Reader
from pipeline.interface import Data
from pipeline.interface import Model
from pipeline.utils import tools
from pipeline.utils.invoker.job_submitter import JobInvoker
from pipeline.utils.logger import LOGGER


class PipeLine(object):
    def __init__(self):
        self._create_time = time.asctime(time.localtime(time.time()))
        self._initiator = None
        self._roles = {}
        self._components = {}
        self._components_input = {}
        self._train_dsl = {}
        self._predict_dsl = {}
        self._train_conf = {}
        self._predict_conf = {}
        self._upload_conf = []
        self._cur_state = None
        self._job_invoker = JobInvoker()
        self._train_job_id = None
        self._predict_job_id = None
        self._fit_status = None
        self._train_board_url = None
        self._model_info = None
        self._train_components = {}
        self._stage = "fit"
        self._data_to_feed_in_prediction = None
        self._predict_pipeline = []
        self._deploy = False

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def set_initiator(self, role, party_id):
        self._initiator = SimpleNamespace(role=role, party_id=party_id)

        return self

    def restore_roles(self, initiator, roles):
        self._initiator = initiator
        self._roles = roles

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def get_predict_meta(self):
        if self._fit_status != StatusCode.SUCCESS:
            raise ValueError("To get predict meta, please fit successfully")

        return {"predict_dsl": self._predict_dsl,
                "train_conf": self._train_conf,
                "initiator": self._initiator,
                "roles": self._roles,
                "model_info": self._model_info,
                "components": self._components,
                "stage": self._stage
                }

    def get_train_dsl(self):
        return copy.deepcopy(self._train_dsl)

    def get_train_conf(self):
        return copy.deepcopy(self._train_conf)

    def get_predict_dsl(self):
        return copy.deepcopy(self._predict_dsl)

    def get_predict_conf(self):
        return copy.deepcopy(self._predict_conf)

    def get_upload_conf(self):
        return copy.deepcopy(self._upload_conf)

    def _get_initiator_conf(self):
        if self._initiator is None:
            raise ValueError("Please set initiator of PipeLine")

        initiator_conf = {"role": self._initiator.role,
                          "party_id": self._initiator.party_id}

        return initiator_conf

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
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

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def add_component(self, component, data=None, model=None):
        if isinstance(component, PipeLine):
            if component.is_deploy() is False:
                raise ValueError("To use a training pipeline object as predict component, should deploy model first")

            if model:
                raise ValueError("pipeline should not have model as input!")

            if not data:
                raise ValueError("To use pipeline as a component, please set data input")

            self._stage = "predict"
            self._predict_pipeline.append({"pipeline": component, "data": data.predict_input})

            meta = component.get_predict_meta()
            self.restore_roles(meta.get("initiator"), meta.get("roles"))

            return

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

                data_key = attr.strip("_")

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

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def add_upload_data(self, file, table_name, namespace, head=1, partition=16, id_delimiter=","):
        data_conf = {"file": file,
                     "table_name": table_name,
                     "namespace": namespace,
                     "head": head,
                     "partition": partition,
                     "id_delimiter": id_delimiter}
        self._upload_conf.append(data_conf)

    def _get_task_inst(self, job_id, name, init_role, party_id):
        return TaskInfo(jobid=job_id,
                        component=self._components[name],
                        job_client=self._job_invoker,
                        role=init_role,
                        party_id=party_id)

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
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

        # print("train_dsl: ", self._train_dsl)
        LOGGER.debug(f"train_dsl: {self._train_dsl}")

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

        # pprint.pprint(self._train_conf)
        LOGGER.debug(f"self._train_conf: \n {json.dumps(self._train_conf, indent=4, ensure_ascii=False)}")
        return self._train_conf

    def _get_job_parameters(self, job_type="train", backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE, version=2):
        job_parameters = {
            "job_type": job_type,
            "backend": backend,
            "work_mode": work_mode,
            "dsl_version": version
        }

        return job_parameters

    def _construct_upload_conf(self, data_conf, backend, work_mode):
        upload_conf = copy.deepcopy(data_conf)
        upload_conf["backend"] = backend
        upload_conf["work_mode"] = work_mode
        return upload_conf

    def describe(self):
        LOGGER.info(f"Pipeline Stage is {self._stage}")
        LOGGER.info("DSL is:")
        if self._stage == "fit":
            LOGGER.info(f"{self._train_dsl}")
        else:
            LOGGER.info(f"{self._predict_dsl}")

        LOGGER.info(f"Pipeline Create Time: {self._create_time}")

    def get_train_job_id(self):
        return self._train_job_id

    def get_predict_job_id(self):
        return self._predict_job_id

    def _set_state(self, state):
        self._cur_state = state

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def compile(self):
        self._construct_train_dsl()
        self._train_conf = self._construct_train_conf()
        if self._stage == "predict":
            predict_pipeline = self._predict_pipeline[0]["pipeline"]
            data_info = self._predict_pipeline[0]["data"]

            meta = predict_pipeline.get_predict_meta()
            if meta["stage"] == "predict":
                raise ValueError(
                    "adding predict pipeline objects'stage is predict, a predict pipeline cannot be an input component")

            self._model_info = meta["model_info"]
            predict_pipeline_dsl = meta["predict_dsl"]
            predict_pipeline_conf = meta["train_conf"]
            if not predict_pipeline_dsl:
                raise ValueError(
                    "Cannot find deploy model in predict pipeline, to use a pipeline as input component, "
                    "it should be deploy first")

            for cpn in self._train_dsl["components"]:
                if cpn in predict_pipeline_dsl["components"]:
                    raise ValueError(
                        "component name {} exist in predict pipeline's deploy component, this is not support")

            if "algorithm_parameters" in predict_pipeline_conf:
                algo_param = predict_pipeline_conf["algorithm_parameters"]
                if "algorithm_parameters" in self._train_conf:
                    for key, value in algo_param.items():
                        if key not in self._train_conf["algorithm_parameters"]:
                            self._train_conf["algorithm_parameters"][key] = value
                else:
                    self._train_conf["algorithm_parameters"] = algo_param

            if "role_parameters" in predict_pipeline_conf:
                role_param = predict_pipeline_conf["role_parameters"]
                for cpn in self._train_dsl["components"]:
                    for role, param in role_param.items():
                        for idx in param:
                            if param[idx].get(cpn) is not None:
                                del predict_pipeline_conf["role_parameters"][role][idx][cpn]

                if "role_parameters" not in self._train_conf:
                    self._train_conf["role_parameters"] = {}

                self._train_conf["role_parameters"] = tools.merge_dict(self._train_conf["role_parameters"],
                                                                       predict_pipeline_conf["role_parameters"])

            self._predict_dsl = tools.merge_dict(predict_pipeline_dsl, self._train_dsl)

            for data_field, val in data_info.items():
                cpn = data_field.split(".", -1)[0]
                dataset = data_field.split(".", -1)[1]
                if not isinstance(val, list):
                    val = [val]

                self._predict_dsl["components"][cpn]["input"]["data"][dataset] = val

    @staticmethod
    def _feed_job_parameters(conf, backend, work_mode, job_type=None, model_info=None):
        submit_conf = copy.deepcopy(conf)
        # print("submit conf' type {}".format(type(submit_conf)))
        LOGGER.debug(f"submit conf type is {type(submit_conf)}")

        #if not isinstance(work_mode, int):
        #    work_mode = work_mode.value
        #if not isinstance(backend, int):
        #    backend = backend.value

        submit_conf["job_parameters"] = {
            "work_mode": work_mode,
            "backend": backend,
            "dsl_version": VERSION
        }

        if job_type is not None:
            submit_conf["job_parameters"]["job_type"] = job_type

        if model_info is not None:
            submit_conf["job_parameters"]["model_id"] = model_info.model_id
            submit_conf["job_parameters"]["model_version"] = model_info.model_version

        return submit_conf

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def fit(self, backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE):
        if self._stage == "predict":
            raise ValueError("This pipeline is constructed for predicting, cannot use fit interface")

        # print("_train_conf {}".format(self._train_conf))
        LOGGER.debug(f"in fit, _train_conf is: \n {json.dumps(self._train_conf)}")
        self._set_state("fit")
        training_conf = self._feed_job_parameters(self._train_conf, backend, work_mode)
        self._train_conf = training_conf
        LOGGER.debug(f"train_conf is: \n {json.dumps(training_conf, indent=4, ensure_ascii=False)}")
        self._train_job_id, detail_info = self._job_invoker.submit_job(self._train_dsl, training_conf)
        self._train_board_url = detail_info["board_url"]
        self._model_info = SimpleNamespace(model_id=detail_info["model_info"]["model_id"],
                                           model_version=detail_info["model_info"]["model_version"])

        self._fit_status = self._job_invoker.monitor_job_status(self._train_job_id,
                                                                self._initiator.role,
                                                                self._initiator.party_id)

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def predict(self, backend=Backend.EGGROLL, work_mode=WorkMode.CLUSTER):
        if self._stage != "predict":
            raise ValueError(
                "To use predict function, please deploy component(s) from training pipeline"
                "and construct a new predict pipeline with data reader and training pipeline.")

        self.compile()

        predict_conf = self._feed_job_parameters(self._train_conf,
                                                 backend,
                                                 work_mode,
                                                 job_type="predict",
                                                 model_info=self._model_info)
        self._predict_conf = copy.deepcopy(predict_conf)
        predict_dsl = copy.deepcopy(self._predict_dsl)

        self._predict_job_id, _ = self._job_invoker.submit_job(dsl=predict_dsl, submit_conf=predict_conf)
        self._job_invoker.monitor_job_status(self._predict_job_id,
                                             self._initiator.role,
                                             self._initiator.party_id)

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def upload(self, backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE, drop=0):
        for data_conf in self._upload_conf:
            upload_conf = self._construct_upload_conf(data_conf, backend, work_mode)
            LOGGER.debug(f"upload_conf is {json.dumps(upload_conf)}")
            self._train_job_id, detail_info = self._job_invoker.upload_data(upload_conf, int(drop))
            self._train_board_url = detail_info["board_url"]
            self._job_invoker.monitor_job_status(self._train_job_id,
                                                 "local",
                                                 0)

    def dump(self, file_path=None):
        pkl = pickle.dumps(self)

        if file_path is not None:
            with open(file_path, "wb") as fout:
                fout.write(pkl)

        return pkl

    @classmethod
    def load(cls, pipeline_bytes):
        return pickle.loads(pipeline_bytes)

    @classmethod
    def load_model_from_file(cls, file_path):
        with open(file_path, "rb") as fin:
            return pickle.loads(fin.read())

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def deploy_component(self, components):
        if self._train_dsl is None:
            raise ValueError("Before deploy model, training should be finish!!!")

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

                if deploy_cpns[-1] not in self._components:
                    raise ValueError("Component {} does not exist in pipeline".format(deploy_cpns[-1]))

                if isinstance(self._components.get(deploy_cpns[-1]), Reader):
                    raise ValueError("Reader should not be include in predict pipeline")

        self._predict_dsl = self._job_invoker.get_predict_dsl(train_dsl=self._train_dsl, cpn_list=deploy_cpns,
                                                              version=VERSION)

        if self._predict_dsl:
            self._deploy = True

        return self

    def is_deploy(self):
        return self._deploy

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def init_predict_config(self, config):
        if isinstance(config, PipeLine):
            config = config.get_predict_meta()

        self._stage = "predict"
        self._model_info = config["model_info"]
        self._predict_dsl = config["predict_dsl"]
        self._train_conf = config["train_conf"]
        self._initiator = config["initiator"]
        self._train_components = config["train_components"]

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def get_component_input_msg(self):
        if VERSION != 2:
            raise ValueError("In DSL Version 1，only need to config data from args, no need special component")

        need_input = {}
        for cpn_name, config in self._predict_dsl["components"].items():
            if "input" not in config:
                continue

            if "data" not in config["input"]:
                continue

            data_config = config["input"]["data"]
            for data_type, dataset_list in data_config.items():
                for data_set in dataset_list:
                    input_cpn = data_set.split(".", -1)[0]
                    input_inst = self._components[input_cpn]
                    if isinstance(input_inst, Reader):
                        if cpn_name not in need_input:
                            need_input[cpn_name] = {}

                        need_input[cpn_name][data_type] = []
                        need_input[cpn_name][data_type].append(input_cpn)

        return need_input

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def get_input_reader_placeholder(self):
        input_info = self.get_component_input_msg()
        input_placeholder = set()
        for cpn_name, data_dict in input_info.items():
            for data_type, dataset_list in data_dict.items():
                for dataset in dataset_list:
                    input_placeholder.add(dataset)

        return input_placeholder

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def set_inputs(self, data_dict):
        if not isinstance(data_dict, dict):
            raise ValueError(
                "inputs for predicting should be a dict, key is input_placeholder name, value is a reader object")

        unfilled_placeholder = self.get_input_reader_placeholder() - set(data_dict.keys())
        if unfilled_placeholder:
            raise ValueError("input placeholder {} should be fill".format(unfilled_placeholder))

        self._data_to_feed_in_prediction = data_dict

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def __getattr__(self, attr):
        if attr in self._components:
            return self._components[attr]

        return self.__getattribute__(attr)

    @LOGGER.catch(onerror=lambda _: sys.exit(1))
    def __getitem__(self, item):
        if item not in self._components:
            raise ValueError("Pipeline does not has component }{}".format(item))

        return self._components[item]

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)
