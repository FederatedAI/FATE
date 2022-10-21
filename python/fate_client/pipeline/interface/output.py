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


class Output(object):
    def __init__(self, name, data_key=None, model_key=None):
        if model_key:
            self.model = Model(name).model
            self.model_output = Model(name).get_all_output()

        if data_key:
            if len(data_key) == 1:
                self.data = SingleOutputData(name, data_key[0]).data
                self.data_output = SingleOutputData(name, data_key[0]).get_all_output()
            else:
                self.data = TraditionalMultiOutputData(name)
                self.data_output = TraditionalMultiOutputData(name).get_all_output()


class Model(object):
    def __init__(self, prefix):
        self.prefix = prefix

    @property
    def model(self):
        return ".".join([self.prefix, "model"])

    @staticmethod
    def get_all_output():
        return ["model"]


class SingleOutputData(object):
    def __init__(self, prefix, data_key):
        self.prefix = prefix
        self._key = data_key

    @property
    def data(self):
        return ".".join([self.prefix, self._key])

    @staticmethod
    def get_all_output():
        return ["data"]


class TraditionalMultiOutputData(object):
    def __init__(self, prefix):
        self.prefix = prefix

    @property
    def train_data(self):
        return ".".join([self.prefix, "train_data"])

    @property
    def test_data(self):
        return ".".join([self.prefix, "test_data"])

    @property
    def validate_data(self):
        return ".".join([self.prefix, "validate_data"])

    @staticmethod
    def get_all_output():
        return ["train_data",
                "validate_data",
                "test_data"]
