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
import getpass
import json
import pickle
import time
from types import SimpleNamespace

from pipeline.backend.config import Role
from pipeline.backend.config import StatusCode
from pipeline.backend.config import VERSION
from pipeline.backend.config import PipelineConfig
from pipeline.backend._operation import OnlineCommand, ModelConvert
from pipeline.backend.task_info import TaskInfo
from pipeline.component.component_base import Component
from pipeline.component.reader import Reader
from pipeline.interface import Data
from pipeline.interface import Model
from pipeline.interface import Cache
from pipeline.utils import tools
from pipeline.utils.invoker.job_submitter import JobInvoker
from pipeline.utils.logger import LOGGER
from pipeline.runtime.entity import JobParameters


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
        self._predict_model_info = None
        self._train_components = {}
        self._stage = "fit"
        self._data_to_feed_in_prediction = None
        self._predict_pipeline = []
        self._deploy = False
        self._system_role = PipelineConfig.SYSTEM_SETTING.get("role")
        self.online = OnlineCommand(self)
        self._load = False
        self.model_convert = ModelConvert(self)
        self._global_job_provider = None

    @LOGGER.catch(reraise=True)
    def set_initiator(self, role, party_id):
        self._initiator = SimpleNamespace(role=role, party_id=party_id)
        # for predict pipeline
        if self._predict_pipeline:
            predict_pipeline = self._predict_pipeline[0]["pipeline"]
            predict_pipeline._initiator = SimpleNamespace(role=role, party_id=party_id)

        return self

    def get_component_list(self):
        return copy.copy(list(self._components.keys()))

    def restore_roles(self, initiator, roles):
        self._initiator = initiator
        self._roles = roles

    @LOGGER.catch(reraise=True)
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

    def get_predict_model_info(self):
        return copy.deepcopy(self._predict_model_info)

    def get_model_info(self):
        return copy.deepcopy(self._model_info)

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

    def set_global_job_provider(self, provider):
        self._global_job_provider = provider
        return self

    @LOGGER.catch(reraise=True)
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
            # update role config for compiled pipeline
            if self._train_conf:
                if role in self._train_conf["role"]:
                    self._train_conf["role"][role] = self._roles[role]

        if self._predict_pipeline:
            predict_pipeline = self._predict_pipeline[0]["pipeline"]
            predict_pipeline._roles = self._roles

        return self

    def _get_role_conf(self):
        return self._roles

    def _get_party_index(self, role, party_id):
        if role not in self._roles:
            raise ValueError("role {} does not setting".format(role))

        if party_id not in self._roles[role]:
            raise ValueError("role {} does not init setting with the party_id {}".format(role, party_id))

        return self._roles[role].index(party_id)

    @LOGGER.catch(reraise=True)
    def add_component(self, component, data=None, model=None, cache=None):
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

            return self

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

        if cache is not None:
            if not isinstance(cache, Cache):
                raise ValueError("cache input of component {} should be passed by cache object".format(component.name))

            attr = cache.cache
            if not isinstance(attr, list):
                attr = [attr]

            self._components_input[component.name]["cache"] = attr

        return self

    @LOGGER.catch(reraise=True)
    def add_upload_data(self, file, table_name, namespace, head=1, partition=16,
                        id_delimiter=",", extend_sid=False, auto_increasing_sid=False, **kargs):
        data_conf = {"file": file,
                     "table_name": table_name,
                     "namespace": namespace,
                     "head": head,
                     "partition": partition,
                     "id_delimiter": id_delimiter,
                     "extend_sid": extend_sid,
                     "auto_increasing_sid": auto_increasing_sid, **kargs}
        self._upload_conf.append(data_conf)

    def _get_task_inst(self, job_id, name, init_role, party_id):
        component = None
        if name in self._components:
            component = self._components[name]

        if component is None:
            if self._stage != "predict":
                raise ValueError(f"Component {name} does not exist")
            training_meta = self._predict_pipeline[0]["pipeline"].get_predict_meta()

            component = training_meta.get("components").get(name)

            if component is None:
                raise ValueError(f"Component {name} does not exist")

        return TaskInfo(jobid=job_id,
                        component=component,
                        job_client=self._job_invoker,
                        role=init_role,
                        party_id=party_id)

    @LOGGER.catch(reraise=True)
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
        if self._global_job_provider:
            self._train_dsl["provider"] = self._global_job_provider

        self._train_dsl["components"] = {}
        for name, component in self._components.items():
            component_dsl = {"module": component.module}
            if name in self._components_input:
                component_dsl["input"] = self._components_input[name]

            if hasattr(component, "output"):
                component_dsl["output"] = {}
                output_attrs = {"data": "data_output",
                                "model": "model_output",
                                "cache": "cache_output"}

                for output_key, attr in output_attrs.items():
                    if hasattr(component.output, attr):
                        component_dsl["output"][output_key] = getattr(component.output, attr)

            provider_name = None
            provider_version = None
            if not hasattr(component, "source_provider"):
                LOGGER.warning(f"Can not retrieval source provider of component {name}, "
                               f"refer to pipeline/component/component_base.py")
            else:
                provider_name = getattr(component, "source_provider")
                if provider_name is None:
                    LOGGER.warning(f"Source provider of component {name} is None, "
                                   f"refer to pipeline/component/component_base.py")

            if hasattr(component, "provider"):
                provider = getattr(component, "provider")
                if provider is not None:
                    if provider.find("@") != -1:
                        provider_name, provider_version = provider.split("@", -1)
                    else:
                        provider_name = provider
                    # component_dsl["provider"] = provider

            if getattr(component, "provider_version") is not None:
                provider_version = getattr(component, "provider_version")

            if provider_name and provider_version:
                component_dsl["provider"] = "@".join([provider_name, provider_version])
            elif provider_name:
                component_dsl["provider"] = provider_name

            self._train_dsl["components"][name] = component_dsl

        if not self._train_dsl:
            raise ValueError("there are no components to train")

        LOGGER.debug(f"train_dsl: {self._train_dsl}")

    def _construct_train_conf(self):
        self._train_conf["dsl_version"] = VERSION
        self._train_conf["initiator"] = self._get_initiator_conf()
        self._train_conf["role"] = self._roles
        self._train_conf["job_parameters"] = {"common": {"job_type": "train"}}
        for name, component in self._components.items():
            param_conf = component.get_config(version=VERSION, roles=self._roles)
            if "common" in param_conf:
                common_param_conf = param_conf["common"]
                if "component_parameters" not in self._train_conf:
                    self._train_conf["component_parameters"] = {}
                if "common" not in self._train_conf["component_parameters"]:
                    self._train_conf["component_parameters"]["common"] = {}

                self._train_conf["component_parameters"]["common"].update(common_param_conf)

            if "role" in param_conf:
                role_param_conf = param_conf["role"]
                if "component_parameters" not in self._train_conf:
                    self._train_conf["component_parameters"] = {}
                if "role" not in self._train_conf["component_parameters"]:
                    self._train_conf["component_parameters"]["role"] = {}
                self._train_conf["component_parameters"]["role"] = tools.merge_dict(
                    role_param_conf, self._train_conf["component_parameters"]["role"])

        LOGGER.debug(f"self._train_conf: \n {json.dumps(self._train_conf, indent=4, ensure_ascii=False)}")
        return self._train_conf

    def _construct_upload_conf(self, data_conf):
        upload_conf = copy.deepcopy(data_conf)
        # upload_conf["work_mode"] = work_mode
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

    def set_job_invoker(self, job_invoker):
        self._job_invoker = job_invoker

    @LOGGER.catch(reraise=True)
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
                        f"component name {cpn} exist in predict pipeline's deploy component, this is not support")

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

                if "input" not in self._predict_dsl["components"][cpn]:
                    self._predict_dsl["components"][cpn]["input"] = {}

                if 'data' not in self._predict_dsl["components"][cpn]["input"]:
                    self._predict_dsl["components"][cpn]["input"]["data"] = {}

                self._predict_dsl["components"][cpn]["input"]["data"][dataset] = val

        return self

    @LOGGER.catch(reraise=True)
    def _check_duplicate_setting(self, submit_conf):
        system_role = self._system_role
        if "role" in submit_conf["job_parameters"]:
            role_conf = submit_conf["job_parameters"]["role"]
            system_role_conf = role_conf.get(system_role, {})
            for party, conf in system_role_conf.items():
                if conf.get("user"):
                    raise ValueError(f"system role {system_role}'s user info already set. Please check.")

    def _feed_job_parameters(self, conf, job_type=None,
                             model_info=None, job_parameters=None):
        submit_conf = copy.deepcopy(conf)
        LOGGER.debug(f"submit conf type is {type(submit_conf)}")

        if job_parameters:
            submit_conf["job_parameters"] = job_parameters.get_config(roles=self._roles)

        if "common" not in submit_conf["job_parameters"]:
            submit_conf["job_parameters"]["common"] = {}

        submit_conf["job_parameters"]["common"]["job_type"] = job_type

        if model_info is not None:
            submit_conf["job_parameters"]["common"]["model_id"] = model_info.model_id
            submit_conf["job_parameters"]["common"]["model_version"] = model_info.model_version

        if self._system_role:
            self._check_duplicate_setting(submit_conf)
            init_role = self._initiator.role
            idx = str(self._roles[init_role].index(self._initiator.party_id))
            if "role" not in submit_conf["job_parameters"]:
                submit_conf["job_parameters"]["role"] = {}

            if init_role not in submit_conf["job_parameters"]["role"]:
                submit_conf["job_parameters"]["role"][init_role] = {}

            if idx not in submit_conf["job_parameters"]["role"][init_role]:
                submit_conf["job_parameters"]["role"][init_role][idx] = {}

            submit_conf["job_parameters"]["role"][init_role][idx].update({"user": getpass.getuser()})
        return submit_conf

    def _filter_out_deploy_component(self, predict_conf):
        if "component_parameters" not in predict_conf:
            return predict_conf

        if "common" in predict_conf["component_parameters"]:
            cpns = list(predict_conf["component_parameters"]["common"])
            for cpn in cpns:
                if cpn not in self._components.keys():
                    del predict_conf["component_parameters"]["common"]

        if "role" in predict_conf["component_parameters"]:
            roles = predict_conf["component_parameters"]["role"].keys()
            for role in roles:
                role_params = predict_conf["component_parameters"]["role"].get(role)
                indexs = role_params.keys()
                for idx in indexs:
                    cpns = role_params[idx].keys()
                    for cpn in cpns:
                        if cpn not in self._components.keys():
                            del role_params[idx][cpn]

                    if not role_params[idx]:
                        del role_params[idx]

                if role_params:
                    predict_conf["component_parameters"]["role"][role] = role_params
                else:
                    del predict_conf["component_parameters"]["role"][role]

        return predict_conf

    @LOGGER.catch(reraise=True)
    def fit(self, job_parameters=None, callback_func=None):

        if self._stage == "predict":
            raise ValueError("This pipeline is constructed for predicting, cannot use fit interface")

        if job_parameters and not isinstance(job_parameters, JobParameters):
            raise ValueError("input parameter of fit function should be JobParameters object")

        LOGGER.debug(f"in fit, _train_conf is: \n {json.dumps(self._train_conf)}")
        self._set_state("fit")
        training_conf = self._feed_job_parameters(self._train_conf, job_type="train", job_parameters=job_parameters)
        self._train_conf = training_conf
        LOGGER.debug(f"train_conf is: \n {json.dumps(training_conf, indent=4, ensure_ascii=False)}")
        self._train_job_id, detail_info = self._job_invoker.submit_job(self._train_dsl, training_conf, callback_func)
        self._train_board_url = detail_info["board_url"]
        self._model_info = SimpleNamespace(model_id=detail_info["model_info"]["model_id"],
                                           model_version=detail_info["model_info"]["model_version"])

        self._fit_status = self._job_invoker.monitor_job_status(self._train_job_id,
                                                                self._initiator.role,
                                                                self._initiator.party_id)

    @LOGGER.catch(reraise=True)
    def update_model_info(self, model_id=None, model_version=None):
        # predict pipeline
        if self._predict_pipeline:
            predict_pipeline = self._predict_pipeline[0]["pipeline"]
            if model_id:
                predict_pipeline._model_info.model_id = model_id
            if model_version:
                predict_pipeline._model_info.model_version = model_version
            return self
        # train pipeline
        original_model_id, original_model_version = None, None
        if self._model_info is not None:
            original_model_id, original_model_version = self._model_info.model_id, self._model_info.model_version
        new_model_id = model_id if model_id is not None else original_model_id
        new_model_version = model_version if model_version is not None else original_model_version
        if new_model_id is None and new_model_version is None:
            return self
        self._model_info = SimpleNamespace(model_id=new_model_id, model_version=new_model_version)
        return self

    @LOGGER.catch(reraise=True)
    def continuously_fit(self):
        self._fit_status = self._job_invoker.monitor_job_status(self._train_job_id,
                                                                self._initiator.role,
                                                                self._initiator.party_id,
                                                                previous_status=self._fit_status)

    @LOGGER.catch(reraise=True)
    def predict(self, job_parameters=None, components_checkpoint=None):
        """

        Parameters
        ----------
        job_parameters: None
        components_checkpoint: specify which model to take, ex.: {"hetero_lr_0": {"step_index": 8}}

        Returns
        -------

        """
        if self._stage != "predict":
            raise ValueError(
                "To use predict function, please deploy component(s) from training pipeline"
                "and construct a new predict pipeline with data reader and training pipeline.")

        if job_parameters and not isinstance(job_parameters, JobParameters):
            raise ValueError("input parameter of fit function should be JobParameters object")

        self.compile()

        res_dict = self._job_invoker.model_deploy(model_id=self._model_info.model_id,
                                                  model_version=self._model_info.model_version,
                                                  predict_dsl=self._predict_dsl,
                                                  components_checkpoint=components_checkpoint)
        self._predict_model_info = SimpleNamespace(model_id=res_dict["model_id"],
                                                   model_version=res_dict["model_version"])
        predict_conf = self._feed_job_parameters(self._train_conf,
                                                 job_type="predict",
                                                 model_info=self._predict_model_info,
                                                 job_parameters=job_parameters)
        predict_conf = self._filter_out_deploy_component(predict_conf)
        self._predict_conf = copy.deepcopy(predict_conf)
        predict_dsl = copy.deepcopy(self._predict_dsl)

        self._predict_job_id, _ = self._job_invoker.submit_job(dsl=predict_dsl, submit_conf=predict_conf)
        self._job_invoker.monitor_job_status(self._predict_job_id,
                                             self._initiator.role,
                                             self._initiator.party_id)

    @LOGGER.catch(reraise=True)
    def upload(self, drop=0):
        for data_conf in self._upload_conf:
            upload_conf = self._construct_upload_conf(data_conf)
            LOGGER.debug(f"upload_conf is {json.dumps(upload_conf)}")
            self._train_job_id, detail_info = self._job_invoker.upload_data(upload_conf, int(drop))
            self._train_board_url = detail_info["board_url"]
            self._job_invoker.monitor_job_status(self._train_job_id,
                                                 "local",
                                                 0)

    @LOGGER.catch(reraise=True)
    def dump(self, file_path=None):
        pkl = pickle.dumps(self)

        if file_path is not None:
            with open(file_path, "wb") as fout:
                fout.write(pkl)

        return pkl

    @classmethod
    def load(cls, pipeline_bytes):
        """
        return pickle.loads(pipeline_bytes)
        """
        pipeline_obj = pickle.loads(pipeline_bytes)
        pipeline_obj.set_job_invoker(JobInvoker())
        return pipeline_obj

    @classmethod
    def load_model_from_file(cls, file_path):
        with open(file_path, "rb") as fin:
            pipeline_obj = pickle.loads(fin.read())
            pipeline_obj.set_job_invoker(JobInvoker())
            return pipeline_obj

    @LOGGER.catch(reraise=True)
    def deploy_component(self, components=None):
        if self._train_dsl is None:
            raise ValueError("Before deploy model, training should be finished!!!")

        if components is None:
            components = self._components
        deploy_cpns = []
        for cpn in components:
            if isinstance(cpn, str):
                deploy_cpns.append(cpn)
            elif isinstance(cpn, Component):
                deploy_cpns.append(cpn.name)
            else:
                raise ValueError(
                    "deploy component parameters is wrong, expect str or Component object, but {} find".format(
                        type(cpn)))

            if deploy_cpns[-1] not in self._components:
                raise ValueError("Component {} does not exist in pipeline".format(deploy_cpns[-1]))

            if isinstance(self._components.get(deploy_cpns[-1]), Reader):
                raise ValueError("Reader should not be include in predict pipeline")

        res_dict = self._job_invoker.model_deploy(model_id=self._model_info.model_id,
                                                  model_version=self._model_info.model_version,
                                                  cpn_list=deploy_cpns)
        self._predict_model_info = SimpleNamespace(model_id=res_dict["model_id"],
                                                   model_version=res_dict["model_version"])

        self._predict_dsl = self._job_invoker.get_predict_dsl(model_id=res_dict["model_id"],
                                                              model_version=res_dict["model_version"])

        if self._predict_dsl:
            self._deploy = True

        return self

    def is_deploy(self):
        return self._deploy

    def is_load(self):
        return self._load

    @LOGGER.catch(reraise=True)
    def init_predict_config(self, config):
        if isinstance(config, PipeLine):
            config = config.get_predict_meta()

        self._stage = "predict"
        self._model_info = config["model_info"]
        self._predict_dsl = config["predict_dsl"]
        self._train_conf = config["train_conf"]
        self._initiator = config["initiator"]
        self._train_components = config["train_components"]

    @LOGGER.catch(reraise=True)
    def get_component_input_msg(self):
        if VERSION != 2:
            raise ValueError("In DSL Version 1，only need to config data from args, do not need special component")

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

    @LOGGER.catch(reraise=True)
    def get_input_reader_placeholder(self):
        input_info = self.get_component_input_msg()
        input_placeholder = set()
        for cpn_name, data_dict in input_info.items():
            for data_type, dataset_list in data_dict.items():
                for dataset in dataset_list:
                    input_placeholder.add(dataset)

        return input_placeholder

    @LOGGER.catch(reraise=True)
    def set_inputs(self, data_dict):
        if not isinstance(data_dict, dict):
            raise ValueError(
                "inputs for predicting should be a dict, key is input_placeholder name, value is a reader object")

        unfilled_placeholder = self.get_input_reader_placeholder() - set(data_dict.keys())
        if unfilled_placeholder:
            raise ValueError("input placeholder {} should be fill".format(unfilled_placeholder))

        self._data_to_feed_in_prediction = data_dict

    @LOGGER.catch(reraise=True)
    def bind_table(self, name, namespace, path, engine='PATH', replace=True, **kwargs):
        info = self._job_invoker.bind_table(engine=engine, name=name, namespace=namespace, address={
            "path": path
        }, drop=replace, **kwargs)
        return info

    # @LOGGER.catch(reraise=True)
    def __getattr__(self, attr):
        if attr in self._components:
            return self._components[attr]

        return self.__getattribute__(attr)

    @LOGGER.catch(reraise=True)
    def __getitem__(self, item):
        if item not in self._components:
            raise ValueError("Pipeline does not has component }{}".format(item))

        return self._components[item]

    def __getstate__(self):
        return vars(self)

    def __setstate__(self, state):
        vars(self).update(state)
