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
import typing
from typing import Literal

import pydantic


class StandaloneComputingSpec(pydantic.BaseModel):
    class MetadataSpec(pydantic.BaseModel):
        computing_id: str
        options: dict = {}

    type: Literal["standalone"]
    metadata: MetadataSpec


class EggrollComputingSpec(pydantic.BaseModel):
    class MetadataSpec(pydantic.BaseModel):
        computing_id: str
        host: typing.Optional[str] = None
        port: typing.Optional[int] = None
        config_options: typing.Optional[dict] = None
        config_properties_file: typing.Optional[str] = None
        options: dict = {}

    type: Literal["eggroll"]
    metadata: MetadataSpec


class SparkComputingSpec(pydantic.BaseModel):
    class MetadataSpec(pydantic.BaseModel):
        computing_id: str

    type: Literal["spark"]
    metadata: MetadataSpec


class CustomComputingSpec(pydantic.BaseModel):
    type: Literal["custom"]
    metadata: dict
