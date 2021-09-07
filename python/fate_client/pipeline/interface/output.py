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


class Output(object):
    def __init__(self, name, data_type='single', has_data=True, has_model=True, has_cache=False, output_unit=1):
        if has_model:
            self.model = Model(name).model
            self.model_output = Model(name).get_all_output()

        if has_data:
            if data_type == "single":
                self.data = SingleOutputData(name).data
                self.data_output = SingleOutputData(name).get_all_output()
            elif data_type == "multi":
                self.data = TraditionalMultiOutputData(name)
                self.data_output = TraditionalMultiOutputData(name).get_all_output()
            else:
                self.data = NoLimitOutputData(name, output_unit)
                self.data_output = NoLimitOutputData(name, output_unit).get_all_output()

        if has_cache:
            self.cache = Cache(name).cache
            self.cache_output = Cache(name).get_all_output()


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
    def __init__(self, prefix):
        self.prefix = prefix

    @property
    def data(self):
        return ".".join([self.prefix, IODataType.SINGLE])

    @staticmethod
    def get_all_output():
        return ["data"]


class TraditionalMultiOutputData(object):
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
    def get_all_output():
        return [IODataType.TRAIN,
                IODataType.VALIDATE,
                IODataType.TEST]


class NoLimitOutputData(object):
    def __init__(self, prefix, output_unit=1):
        self.prefix = prefix
        self.output_unit = output_unit

    @property
    def data(self):
        return [self.prefix + "." + "data_" + str(i) for i in range(self.output_unit)]

    def get_all_output(self):
        return ["data_" + str(i) for i in range(self.output_unit)]


class Cache(object):
    def __init__(self, prefix):
        self.prefix = prefix

    @property
    def cache(self):
        return ".".join([self.prefix, "cache"])

    @staticmethod
    def get_all_output():
        return ["cache"]
