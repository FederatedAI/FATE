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
from fate_arch.common import WorkMode, Backend
from enum import IntEnum


class RunParameters(object):
    def __init__(self, **kwargs):
        self.job_type = "train"
        self.work_mode = WorkMode.STANDALONE
        self.backend = Backend.EGGROLL  # Pre-v1.5 configuration item
        self.computing_engine = None
        self.federation_engine = None
        self.storage_engine = None
        self.engines_address = {}
        self.federated_mode = None
        self.federation_info = None
        self.task_parallelism = None
        self.task_nodes = None
        self.task_cores_per_node = None
        self.task_memory_per_node = None
        self.federated_status_collect_type = None
        self.federated_data_exchange_type = None  # not use in v1.5.0
        self.align_task_input_data_partition = None
        self.model_id = None
        self.model_version = None
        self.dsl_version = None
        self.input_data_aligned_partitions = None

        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    def to_dict(self):
        d = {}
        for k, v in self.__dict__.items():
            if v is None:
                continue
            d[k] = v
        return d


class RetCode(IntEnum):
    SUCCESS = 0
    EXCEPTION_ERROR = 100
    PARAMETER_ERROR = 101
    DATA_ERROR = 102
    OPERATING_ERROR = 103
    FEDERATED_ERROR = 104
    CONNECTION_ERROR = 105
    SERVER_ERROR = 500


class SchedulingStatusCode(object):
    SUCCESS = 0
    NO_RESOURCE = 1
    PASS = 1
    NO_NEXT = 2
    HAVE_NEXT = 3
    FAILED = 4


class FederatedSchedulingStatusCode(object):
    SUCCESS = 0
    PARTIAL = 1
    FAILED = 2


class BaseStatus(object):
    @classmethod
    def status_list(cls):
        return [cls.__dict__[k] for k in cls.__dict__.keys() if not callable(getattr(cls, k)) and not k.startswith("__")]

    @classmethod
    def contains(cls, status):
        return status in cls.status_list()


class StatusSet(BaseStatus):
    WAITING = 'waiting'
    READY = 'ready'
    RUNNING = "running"
    CANCELED = "canceled"
    TIMEOUT = "timeout"
    FAILED = "failed"
    COMPLETE = "complete"

    @classmethod
    def get_level(cls, status):
        return dict(zip(cls.status_list(), range(len(cls.status_list())))).get(status, None)


class BaseStateTransitionRule(object):
    RULES = {}

    @classmethod
    def if_pass(cls, src_status, dest_status):
        if src_status not in cls.RULES:
            return False
        if dest_status not in cls.RULES[src_status]:
            return False
        else:
            return True


class JobStatus(BaseStatus):
    WAITING = StatusSet.WAITING
    READY = StatusSet.READY
    RUNNING = StatusSet.RUNNING
    CANCELED = StatusSet.CANCELED
    TIMEOUT = StatusSet.TIMEOUT
    FAILED = StatusSet.FAILED
    COMPLETE = StatusSet.COMPLETE

    class StateTransitionRule(BaseStateTransitionRule):
        RULES = {
            StatusSet.WAITING: [StatusSet.READY, StatusSet.RUNNING, StatusSet.CANCELED, StatusSet.TIMEOUT, StatusSet.FAILED, StatusSet.COMPLETE],
            StatusSet.READY: [StatusSet.WAITING, StatusSet.RUNNING],
            StatusSet.RUNNING: [StatusSet.CANCELED, StatusSet.TIMEOUT, StatusSet.FAILED, StatusSet.COMPLETE],
            StatusSet.CANCELED: [StatusSet.WAITING],
            StatusSet.TIMEOUT: [StatusSet.FAILED, StatusSet.COMPLETE, StatusSet.WAITING],
            StatusSet.FAILED: [StatusSet.WAITING],
            StatusSet.COMPLETE: [StatusSet.WAITING],
        }


class TaskStatus(BaseStatus):
    WAITING = StatusSet.WAITING
    RUNNING = StatusSet.RUNNING
    CANCELED = StatusSet.CANCELED
    TIMEOUT = StatusSet.TIMEOUT
    FAILED = StatusSet.FAILED
    COMPLETE = StatusSet.COMPLETE

    class StateTransitionRule(BaseStateTransitionRule):
        RULES = {
            StatusSet.WAITING: [StatusSet.RUNNING, StatusSet.CANCELED, StatusSet.TIMEOUT, StatusSet.FAILED],
            StatusSet.RUNNING: [StatusSet.CANCELED, StatusSet.TIMEOUT, StatusSet.FAILED, StatusSet.COMPLETE],
            StatusSet.CANCELED: [],
            StatusSet.TIMEOUT: [StatusSet.FAILED, StatusSet.COMPLETE],
            StatusSet.FAILED: [],
            StatusSet.COMPLETE: [],
        }


class OngoingStatus(BaseStatus):
    WAITING = StatusSet.WAITING
    RUNNING = StatusSet.RUNNING


class InterruptStatus(BaseStatus):
    CANCELED = StatusSet.CANCELED
    TIMEOUT = StatusSet.TIMEOUT
    FAILED = StatusSet.FAILED


class EndStatus(BaseStatus):
    CANCELED = StatusSet.CANCELED
    TIMEOUT = StatusSet.TIMEOUT
    FAILED = StatusSet.FAILED
    COMPLETE = StatusSet.COMPLETE


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


class KillProcessStatusCode(object):
    KILLED = 0
    NOT_FOUND = 1
    ERROR_PID = 2
