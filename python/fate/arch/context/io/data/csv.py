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


class CSVReader:
    def __init__(self, ctx, path, metadata: dict) -> None:
        self.ctx = ctx
        self.path = path
        self.metadata = metadata

    def read_dataframe(self):
        import inspect

        from fate.arch import dataframe

        kwargs = {}
        p = inspect.signature(dataframe.CSVReader.__init__).parameters
        parameter_keys = p.keys()
        for k, v in self.metadata.items():
            if k in parameter_keys:
                kwargs[k] = v

        return dataframe.CSVReader(**kwargs).to_frame(self.ctx, self.path)


class CSVWriter:
    def __init__(self, ctx, path, metadata: dict) -> None:
        self.ctx = ctx
        self.path = path
        self.metadata = metadata

    def write_dataframe(self, df):
        ...
