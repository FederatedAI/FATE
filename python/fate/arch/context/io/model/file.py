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
import json

from ....unify import URI


class FileModelWriter:
    def __init__(self, ctx, name: str, uri: URI) -> None:
        self.ctx = ctx
        self.name = name
        self.path = uri.path

    def write_model(self, model):
        with open(self.path, "w") as f:
            json.dump(model, f)


class FileModelReader:
    def __init__(self, ctx, name: str, uri: URI, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = uri
        self.metadata = metadata

    def read_model(self):
        with open(self.uri.path, "r") as fin:
            return json.loads(fin.read())
