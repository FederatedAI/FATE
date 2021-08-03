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
import os
from fate_arch.common import file_utils
from fate_flow.entity.types import ComponentProviderName


class ComponentProvider(object):
    def __init__(self, name, version, path, **kwargs):
        if not ComponentProviderName.contains(name):
            raise ValueError(f"not support {name} provider")
        self._name = name
        self._version = version
        self._path = path
        self._env = {}
        self.init_env()

    def init_env(self):
        self._env["PYTHONPATH"] = os.path.join(file_utils.get_python_base_directory(), *self._path[:-1])

    @property
    def name(self):
        return self._name

    @property
    def version(self):
        return self._version

    @property
    def path(self):
        return self._path

    @property
    def env(self):
        return self._env

    def to_json(self):
        return {k.lstrip('_'): v for k, v in self.__dict__.items()}