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
from pipeline.backend.config import IODataType


class Input(object):
    def __init__(self, name, data_type="single"):
        if data_type == "single":
            self.data = InputData(name).data
            self.data_output = InputData(name).get_all_input()
        elif data_type == "multi":
            self.data = TrainingInputData(name)
            self.data_output = InputData(name).get_all_input()
        else:
            raise ValueError("input data type should be one of ['single', 'multi']")


class InputData(object):
    def __init__(self, prefix):
        self.prefix = prefix

    @property
    def data(self):
        return ".".join([self.prefix, IODataType.SINGLE])

    @staticmethod
    def get_all_input():
        return ["data"]


class TrainingInputData(object):
    def __init__(self, prefix):
        self.prefix = prefix

    @property
    def train_data(self):
        return ".".join([self.prefix, IODataType.TRAIN])

    @property
    def test_data(self):
        return ".".join([self.prefix, IODataType.TEST])

    @property
    def validate_data(self):
        return ".".join([self.prefix, IODataType.VALIDATE])

    @staticmethod
    def get_all_input():
        return [IODataType.TRAIN,
                IODataType.VALIDATE,
                IODataType.TEST]
