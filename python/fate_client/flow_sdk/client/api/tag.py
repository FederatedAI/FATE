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
from flow_sdk.utils import preprocess


class Tag(BaseFlowAPI):
    def create(self, tag_name, tag_desc=None):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/tag/create', json=config_data)

    def query(self, tag_name, with_model=False):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/tag/retrieve', json=config_data)

    def update(self, tag_name, new_tag_name=None, new_tag_desc=None):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/tag/update', json=config_data)

    def delete(self, tag_name):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/tag/destroy', json=config_data)

    def list(self, limit=10):
        kwargs = locals()
        config_data, dsl_data = preprocess(**kwargs)
        return self._post(url='model/tag/list', json=config_data)
