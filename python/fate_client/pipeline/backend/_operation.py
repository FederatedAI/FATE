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


# from pipeline.backend.config import WorkMode
from pipeline.utils.logger import LOGGER


class OnlineCommand(object):
    def __init__(self, pipeline_obj):
        self.pipeline_obj = pipeline_obj

    """
    def _feed_online_conf(self):
        conf = {"initiator": self.pipeline_obj._get_initiator_conf(),
                "role": self.pipeline_obj._roles}
        predict_model_info = self.pipeline_obj.get_predict_model_info()
        train_work_mode = self.pipeline_obj.get_train_conf().get("job_parameters").get("common").get("work_mode")
        if train_work_mode != WorkMode.CLUSTER:
            raise ValueError(f"to use FATE serving online inference service, work mode must be CLUSTER.")
        conf["job_parameters"] = {"model_id": predict_model_info.model_id,
                                  "model_version": predict_model_info.model_version,
                                  "work_mode": WorkMode.CLUSTER}
        return conf
    """

    def _feed_online_conf(self):
        conf = {"initiator": self.pipeline_obj._get_initiator_conf(),
                "role": self.pipeline_obj._roles}
        predict_model_info = self.pipeline_obj.get_predict_model_info()
        conf["job_parameters"] = {"model_id": predict_model_info.model_id,
                                  "model_version": predict_model_info.model_version}
        return conf

    @LOGGER.catch(reraise=True)
    def load(self, file_path=None):
        if not self.pipeline_obj.is_deploy():
            raise ValueError(f"to load model for online inference, must deploy components first.")
        file_path = file_path if file_path else ""
        load_conf = self._feed_online_conf()
        load_conf["job_parameters"]["file_path"] = file_path
        self.pipeline_obj._job_invoker.load_model(load_conf)
        self.pipeline_obj._load = True

    @LOGGER.catch(reraise=True)
    def bind(self, service_id, *servings):
        if not self.pipeline_obj.is_deploy() or not self.pipeline_obj.is_load():
            raise ValueError(f"to bind model to online service, must deploy and load model first.")
        bind_conf = self._feed_online_conf()
        bind_conf["service_id"] = service_id
        bind_conf["servings"] = list(servings)
        self.pipeline_obj._job_invoker.bind_model(bind_conf)


class ModelConvert(object):
    def __init__(self, pipeline_obj):
        self.pipeline_obj = pipeline_obj

    def _feed_homo_conf(self, framework_name):
        model_info = self.pipeline_obj.get_model_info()
        conf = {"role": self.pipeline_obj._initiator.role,
                "party_id": self.pipeline_obj._initiator.party_id,
                "model_id": model_info.model_id,
                "model_version": model_info.model_version
                }
        if framework_name:
            conf["framework_name"] = framework_name
        return conf

    @LOGGER.catch(reraise=True)
    def convert(self, framework_name=None):
        if self.pipeline_obj._train_dsl is None:
            raise ValueError("Before converting homo model, training should be finished!!!")
        conf = self._feed_homo_conf(framework_name)
        res_dict = self.pipeline_obj._job_invoker.convert_homo_model(conf)
        return res_dict
