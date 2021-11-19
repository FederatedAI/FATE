#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
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
from flow_sdk.client.api.base import BaseFlowAPI
from flow_sdk.utils import preprocess, check_config


class Checkpoint(BaseFlowAPI):

    def list_checkpoints(self, **kwargs):
        config_data, dsl_data = preprocess(**kwargs)
        check_config(config=config_data,
                     required_arguments=['role', 'party_id', 'model_id', 'model_version', 'component_name'])
        return self._post(url='checkpoint/list', json=config_data)

    def get_checkpoint(self, **kwargs):
        config_data, dsl_data = preprocess(**kwargs)
        check_config(config=config_data,
                     required_arguments=['role', 'party_id', 'model_id', 'model_version', 'component_name'])
        if len(config_data.keys() & {'step_index', 'step_name'}) != 1:
            raise KeyError('step_index or step_name is required')
        return self._post(url='checkpoint/get', json=config_data)
