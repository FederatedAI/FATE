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
import _md5


class NamingService(object):
    __instance = None

    @classmethod
    def get_instance(cls):
        if cls.__instance is None:
            raise EnvironmentError("naming service not set")
        return cls.__instance

    @classmethod
    def set_instance(cls, instance):
        prev = cls.__instance
        cls.__instance = instance
        return prev

    def __init__(self, init_name="ss"):
        self._name = _md5.md5(init_name.encode("utf-8")).hexdigest()

    def next(self):
        self._name = _md5.md5(self._name.encode("utf-8")).hexdigest()
        return self._name
