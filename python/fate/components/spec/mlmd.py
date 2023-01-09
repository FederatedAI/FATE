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
from typing import Literal

from pydantic import BaseModel


class PipelineMLMDSpec(BaseModel):
    class PipelineMLMDMetaData(BaseModel):
        db: str

    type: Literal["pipeline"]
    metadata: PipelineMLMDMetaData


class FlowMLMDSpec(BaseModel):
    class FlowMLMDMetaData(BaseModel):
        statu_uri: str
        tracking_uri: str

    type: Literal["flow"]
    metadata: FlowMLMDMetaData


class NoopMLMDSpec(BaseModel):
    type: Literal["noop"]


class CustomMLMDSpec(BaseModel):
    class CustomMLMDMetaData(BaseModel):
        entrypoint: str

    type: Literal["custom"]
    name: str
    metadata: CustomMLMDMetaData
