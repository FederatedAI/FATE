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
from typing import Protocol


class ExecutionStatus(Protocol):
    def log_excution_start(self):
        ...

    def log_excution_end(self):
        ...

    def log_excution_exception(self, message: dict):
        ...

    def safe_terminate(self):
        ...


class IOManagerProtocol:
    def log_input_parameter(self, key, value):
        ...

    def log_input_artifact(self, key, value):
        ...

    def log_output_artifact(self, key, value):
        ...

    def log_output_data(self, key, value):
        ...

    def log_output_model(self, key, value, metadata={}):
        ...

    def log_output_metric(self, key, value):
        ...


class MLMD(Protocol):
    execution_status: ExecutionStatus
    io: IOManagerProtocol
