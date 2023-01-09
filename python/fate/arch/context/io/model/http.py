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
import logging

import requests

from ....unify import URI

logger = logging.getLogger(__name__)


class HTTPModelWriter:
    def __init__(self, ctx, name: str, uri: URI, metadata) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = uri
        self.entrypoint = f"{self.uri.schema}://{self.uri.authority}{self.uri.path}"

    def write_model(self, model):
        logger.debug(self.entrypoint)
        response = requests.post(url=self.entrypoint, json={"data": model})
        logger.debug(response.text)


class HTTPModelReader:
    def __init__(self, ctx, name: str, uri: URI, metadata: dict) -> None:
        self.name = name
        self.ctx = ctx
        self.uri = uri
        self.entrypoint = f"{self.uri.schema}://{self.uri.authority}{self.uri.path}"
        self.metadata = metadata

    def read_model(self):
        return requests.get(url=self.entrypoint).json().get("data", {})
