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
from flow_sdk.client.api.base import BaseFlowAPI
from flow_sdk.utils import preprocess, check_config


class Privilege(BaseFlowAPI):
    def query(self, src_party_id, src_role):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        check_config(config=config_data, required_arguments=['src_party_id', 'src_role'])
        return self._post(url='permission/query/privilege', json=config_data)

    def grant(self, src_party_id, src_role):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        check_config(config=config_data, required_arguments=['src_party_id', 'src_role'])
        return self._post(url='permission/grant/privilege', json=config_data)

    def delete(self, src_party_id, src_role):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        check_config(config=config_data, required_arguments=['src_party_id', 'src_role'])
        return self._post(url='permission/delete/privilege', json=config_data)
