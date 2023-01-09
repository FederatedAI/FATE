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
from ....unify import FileURI
from .df import Dataframe


class FileDataFrameWriter:
    def __init__(self, ctx, name: str, uri: FileURI, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = FileMetaURI(uri)
        self.metadata = metadata

    def write_dataframe(self, df):
        import json

        from fate.arch import dataframe

        table = dataframe.serialize(self.ctx, df)
        with open(self.uri.get_data_path(), "w") as f:
            json.dump(list(table.collect()), f)
        with open(self.uri.get_meta_path(), "w") as f:
            json.dump(table.schema, f)


class FileDataFrameReader:
    def __init__(self, ctx, name: str, uri: FileURI, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = FileMetaURI(uri)
        self.metadata = metadata

    def read_dataframe(self):
        import json

        from fate.arch import dataframe

        with open(self.uri.get_meta_path(), "r") as fin:
            schema = json.load(fin)
        with open(self.uri.get_data_path(), "r") as fin:
            data = json.load(fin)

        table = self.ctx.computing.parallelize(data, include_key=True, partition=1)
        table.schema = schema
        df = dataframe.deserialize(self.ctx, table)

        return Dataframe(df, df.shape[1], df.shape[0])


class FileMetaURI:
    def __init__(self, uri: FileURI) -> None:
        self.uri = uri

    def get_data_path(self):
        return self.uri.path

    def get_meta_path(self):
        return f"{self.uri.path}.meta"
