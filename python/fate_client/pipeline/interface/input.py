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
from ..conf.types import InputDataKey
from ..conf.types import InputModelKey
from ..conf.types import InputCacheKey


class Input(object):
    def __init__(self, name, data_key=None, model_key=None, cache_key=None):
        self._data_key = InputDataKey().register(data_key) if data_key else None
        self._model_key = InputModelKey().register(model_key) if model_key else None
        self._cache_key = InputCacheKey().register(cache_key) if cache_key else None

    def get_input_keys(self):
        inputs = dict()
        if self._data_key:
            inputs["data"] = self._data_key.keys

        if self._model_key:
            inputs["model"] = self._model_key

        if self._cache_key:
            inputs["cache"] = self._cache_key

        return inputs
