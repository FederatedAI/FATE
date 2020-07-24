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


class RetCode(IntEnum):
    SUCCESS = 0
    EXCEPTION_ERROR = 100
    PARAM_ERROR = 101
    DATA_ERROR = 102
    REPORT_ERROR = 103
    FEDERATED_ERROR = 104
    CONNECTION_ERROR = 105
    SERVER_ERROR = 500


class FederatedSchedulingStatusCode(object):
    SUCCESS = 0
    PARTIAL = 1
    FAILED = 3


class BaseJobStatus(object):
    WAITING = 'WAITING'
    START = 'START'
    RUNNING = "RUNNING"
    CANCELED = "CANCELED"
    TIMEOUT = "TIMEOUT"
    FAILED = "FAILED"
    COMPLETE = "COMPLETE"

    @staticmethod
    def get_level(status):
        return dict(zip(BaseJobStatus.__dict__.keys(), range(len(BaseJobStatus.__dict__.keys())))).get(status, None)


class JobStatus(object):
    WAITING = BaseJobStatus.WAITING
    RUNNING = BaseJobStatus.RUNNING
    CANCELED = BaseJobStatus.CANCELED
    TIMEOUT = BaseJobStatus.TIMEOUT
    FAILED = BaseJobStatus.FAILED
    COMPLETE = BaseJobStatus.COMPLETE


class TaskSetStatus(object):
    WAITING = BaseJobStatus.WAITING
    RUNNING = BaseJobStatus.RUNNING
    CANCELED = BaseJobStatus.CANCELED
    TIMEOUT = BaseJobStatus.TIMEOUT
    FAILED = BaseJobStatus.FAILED
    COMPLETE = BaseJobStatus.COMPLETE


class TaskStatus(object):
    WAITING = BaseJobStatus.WAITING
    START = BaseJobStatus.START
    RUNNING = BaseJobStatus.RUNNING
    CANCELED = BaseJobStatus.CANCELED
    TIMEOUT = BaseJobStatus.TIMEOUT
    FAILED = BaseJobStatus.FAILED
    COMPLETE = BaseJobStatus.COMPLETE


class EndStatus(object):
    CANCELED = BaseJobStatus.CANCELED
    TIMEOUT = BaseJobStatus.TIMEOUT
    FAILED = BaseJobStatus.FAILED
    COMPLETE = BaseJobStatus.COMPLETE

    @staticmethod
    def is_end_status(status):
        return status in EndStatus.__dict__.keys()

    @staticmethod
    def status_list():
        return EndStatus.__dict__.keys()


class InterruptStatus(object):
    CANCELED = BaseJobStatus.CANCELED
    TIMEOUT = BaseJobStatus.TIMEOUT
    FAILED = BaseJobStatus.FAILED

    @staticmethod
    def is_interrupt_status(status):
        return status in InterruptStatus.__dict__.keys()

    @staticmethod
    def status_list():
        return InterruptStatus.__dict__.keys()


class ModelStorage(object):
    REDIS = "redis"
    MYSQL = "mysql"


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