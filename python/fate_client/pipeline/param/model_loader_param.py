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

from pipeline.param.base_param import BaseParam


class CheckpointParam(BaseParam):

    def __init__(self, model_id: str = None, model_version: str = None, component_name: str = None,
                 step_index: int = None, step_name: str = None):
        super().__init__()
        self.model_id = model_id
        self.model_version = model_version
        self.component_name = component_name
        self.step_index = step_index
        self.step_name = step_name

        if self.step_index is not None:
            self.step_index = int(self.step_index)

    def check(self):
        for i in ('model_id', 'model_version', 'component_name'):
            if getattr(self, i) is None:
                return False

        # do not set step_index and step_name at the same time
        if self.step_index is not None:
            return self.step_name is None
        return self.step_name is not None
