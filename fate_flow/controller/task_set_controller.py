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
import threading
import time
import sys

from fate_flow.utils.authentication_utils import authentication_check
from federatedml.protobuf.generated import pipeline_pb2
from arch.api.utils.log_utils import schedule_logger
from fate_flow.db.db_models import Task
from fate_flow.operation.task_executor import TaskExecutor
from fate_flow.scheduler.task_scheduler import TaskScheduler
from fate_flow.scheduler.federated_scheduler import FederatedScheduler
from fate_flow.entity.constant import JobStatus, TaskSetStatus, TaskStatus
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.operation.job_tracker import Tracker
from fate_flow.settings import USE_AUTHENTICATION
from fate_flow.utils import job_utils, job_controller_utils
from fate_flow.utils.job_utils import save_job_conf, get_job_dsl_parser
import os
from fate_flow.operation.job_saver import JobSaver
from arch.api.utils.core_utils import json_dumps
from fate_flow.entity.constant import Backend
from fate_flow.controller.task_controller import TaskController


class TaskSetController(object):
    @classmethod
    def update_task_set(cls, task_set_info):
        """
        Save to local database
        :param task_set_info:
        :return:
        """
        JobSaver.update_task_set(task_set_info=task_set_info)

    @classmethod
    def stop_task_set(cls, task_set, stop_status):
        """
        Stop all tasks in taskSet, including all task versions
        :param task_set:
        :param stop_status:
        :return:
        """
        tasks = job_utils.query_task(job_id=task_set.f_job_id, task_set_id=task_set.f_task_set_id, role=task_set.f_role, party_id=task_set.f_party_id)
        for task in tasks:
            TaskController.stop_task(task=task, stop_status=stop_status)
        # TaskSet status depends on the final operation result and initiator calculate
