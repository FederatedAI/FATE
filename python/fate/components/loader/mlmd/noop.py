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
from fate.components.spec.mlmd import NoopMLMDSpec

from .protocol import MLMD
from .protocol import ExecutionStatus as ExecutionStatusProtocol
from .protocol import IOManagerProtocol


class NoopMLMD(MLMD):
    def __init__(self, mlmd: NoopMLMDSpec, taskid) -> None:
        self._taskid = taskid
        self.execution_status = ExecutionStatus()
        self.io = IOManager()


class IOManager(IOManagerProtocol):
    def __init__(self) -> None:
        ...

    def log_input_artifact(self, key, value):
        print(f"log input artifact: {key}, {value}")

    def log_output_artifact(self, key, value):
        print(f"log output artifact: {key}, {value}")

    def log_input_parameter(self, key, value):
        print(f"log input parameter: {key}, {value}")

    def log_input_data(self, key, value):
        print(f"log input data: {key}, {value}")

    def log_input_model(self, key, value):
        print(f"log input model: {key}, {value}")

    def log_input_metric(self, key, value):
        print(f"log input metric: {key}, {value}")

    def log_output_data(self, key, value):
        print(f"log output data: {key}, {value}")

    def log_output_model(self, key, value, metadata={}):
        print(f"log output model: {key}, {value}, {metadata}")

    def log_output_metric(self, key, value):
        print(f"log output metric: {key}, {value}")


class ExecutionStatus(ExecutionStatusProtocol):
    def __init__(self) -> None:
        ...

    def log_excution_start(self):
        print(f"running")

    def log_excution_end(self):
        print(f"end")

    def log_excution_exception(self, message: dict):
        print(f"exception: {message}")

    def safe_terminate(self):
        return True
