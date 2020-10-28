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
from fate_arch.common import log

LOGGER = log.getLogger()


class ComponentBase(object):
    def __init__(self):
        self.task_version_id = ''
        self.tracker = None
        self.model_output = None
        self.data_output = None

    def run(self, component_parameters: dict = None, run_args: dict = None):
        pass

    def set_tracker(self, tracker):
        self.tracker = tracker

    def save_data(self):
        return self.data_output

    def export_model(self):
        return self.model_output

    def set_task_version_id(self, task_version_id):
        self.task_version_id = task_version_id