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
import datetime
from arch.task_manager.settings import PARTY_ID
from arch.api.utils import file_utils
import threading
import os
from flask import jsonify


class IdCounter:
    _lock = threading.RLock()

    def __init__(self, initial_value=0):
        self._value = initial_value

    def incr(self, delta=1):
        '''
        Increment the counter with locking
        '''
        with IdCounter._lock:
            self._value += delta
            return self._value


id_counter = IdCounter()


def generate_job_id():
    return '_'.join([datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), str(PARTY_ID), str(id_counter.incr())])


def get_job_directory(job_id=None):
    _paths = ['jobs', job_id] if job_id else ['jobs']
    return os.path.join(file_utils.get_project_base_directory(), *_paths)


def get_json_result(status=0, msg='success'):
    return jsonify({"status": status, "msg": msg})
