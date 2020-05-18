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


class WorkMode(IntEnum):
    STANDALONE = 0
    CLUSTER = 1


class Backend(IntEnum):
    EGGROLL = 0
    SPARK = 1

    def is_eggroll(self):
        return self.value == self.EGGROLL

    def is_spark(self):
        return self.value == self.SPARK


class JobStatus(object):
    WAITING = 'waiting'
    RUNNING = 'running'
    COMPLETE = 'success'
    FAILED = 'failed'
    TIMEOUT = 'timeout'
    CANCELED = 'canceled'
    PARTIAL = 'partial'
    DELETED = 'deleted'


class TaskStatus(object):
    START = 'start'
    RUNNING = 'running'
    COMPLETE = 'success'
    FAILED = 'failed'
    TIMEOUT = 'timeout'


class ModelStorage(object):
    REDIS = "redis"


class ModelOperation(object):
    EXPORT = "export"
    IMPORT = "import"
    STORE = "store"
    RESTORE = "restore"
    LOAD = "load"
    BIND = "bind"


class ProcessRole(object):
    SERVER = "server"
    EXECUTOR = "executor"