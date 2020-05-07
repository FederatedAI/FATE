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
from arch.api.utils import log_utils
from fate_flow.entity.constant_config import ModelStorage
from fate_flow.manager.model_manager import redis_model_storage
from fate_flow.components.component_base import ComponentBase

LOGGER = log_utils.getLogger()


ModelStorageClassMap = {
    ModelStorage.REDIS: redis_model_storage.RedisModelStorage
}


class ModelStore(ComponentBase):
    def run(self, component_parameters: dict = None, run_args: dict = None):
        parameters = component_parameters.get("ModelStoreParam", dict)
        model_storage = ModelStorageClassMap.get(parameters["store_address"]["storage"])()
        model_storage.store(model_id=parameters["model_id"],
                            model_version=parameters["model_version"],
                            store_address=parameters["store_address"],
                            )


class ModelRestore(ComponentBase):
    def run(self, component_parameters: dict = None, run_args: dict = None):
        parameters = component_parameters.get("ModelRestoreParam", dict)
        model_storage = ModelStorageClassMap.get(parameters["store_address"]["storage"])()
        model_storage.restore(model_id=parameters["model_id"],
                              model_version=parameters["model_version"],
                              store_address=parameters["store_address"],
                              )
