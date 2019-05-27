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

from arch.api.utils import cloudpickle
from abc import ABCMeta
from abc import abstractmethod
from pickle import loads as p_loads
from pickle import dumps as p_dumps


class ABCSerdes:
    __metaclass__ = ABCMeta

    @staticmethod
    @abstractmethod
    def serialize(_obj):
        pass

    @staticmethod
    @abstractmethod
    def deserialize(_bytes):
        pass


class CloudPickleSerdes(ABCSerdes):

    @staticmethod
    def serialize(_obj):
        return cloudpickle.dumps(_obj)

    @staticmethod
    def deserialize(_bytes):
        return cloudpickle.loads(_bytes)


class PickleSerdes(ABCSerdes):

    @staticmethod
    def serialize(_obj):
        return p_dumps(_obj)

    @staticmethod
    def deserialize(_bytes):
        return p_loads(_bytes)


serdes_cache = {}
for cls in ABCSerdes.__subclasses__():
    cls_name = ".".join([cls.__module__, cls.__qualname__])
    serdes_cache[cls_name] = cls


def get_serdes(serdes_id=None):
    try:
        return serdes_cache[serdes_id]
    except:
        return PickleSerdes
