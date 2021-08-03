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
from enum import IntEnum


class BaseType(object):
    @classmethod
    def types(cls):
        return [cls.__dict__[k] for k in cls.__dict__.keys() if not callable(getattr(cls, k)) and not k.startswith("__")]

    @classmethod
    def contains(cls, status):
        return status in cls.types()


class ComponentProviderName(BaseType):
    FATE_FEDERATED_ALGORITHM = "fate_federated_algorithm"
    FUSHU_AVATAR_ALGORITHM = "fushu_avatar_algorithm"
    FATE_FLOW_TOOLS = "fate_flow_tools"


class RetCode(IntEnum):
    SUCCESS = 0
    EXCEPTION_ERROR = 100
    PARAMETER_ERROR = 101
    DATA_ERROR = 102
    OPERATING_ERROR = 103
    FEDERATED_ERROR = 104
    CONNECTION_ERROR = 105
    SERVER_ERROR = 500


class ModelStorage(object):
    REDIS = "redis"
    MYSQL = "mysql"


class ModelOperation(object):
    STORE = "store"
    RESTORE = "restore"
    EXPORT = "export"
    IMPORT = "import"
    LOAD = "load"
    BIND = "bind"


class ProcessRole(object):
    DRIVER = "driver"
    EXECUTOR = "executor"


class TagOperation(object):
    CREATE = "create"
    RETRIEVE = "retrieve"
    UPDATE = "update"
    DESTROY = "destroy"
    LIST = "list"


class ResourceOperation(object):
    APPLY = "apply"
    RETURN = "return"


class KillProcessRetCode(object):
    KILLED = 0
    NOT_FOUND = 1
    ERROR_PID = 2


class InputSearchType(IntEnum):
    UNKNOWN = 0
    TABLE_INFO = 1
    JOB_COMPONENT_OUTPUT = 2
