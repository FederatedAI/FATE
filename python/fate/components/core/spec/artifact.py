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
import re
from typing import Dict, List, Optional, Union

import pydantic

from .model import (
    MLModelSpec,
)

# see https://www.rfc-editor.org/rfc/rfc3986#appendix-B
# scheme    = $2
# authority = $4
# path      = $5
# query     = $7
# fragment  = $9
_uri_regex = re.compile(r"^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?")


class DataOverview(pydantic.BaseModel):
    count: Optional[int] = None
    samples: Optional[List] = None


class ArtifactSource(pydantic.BaseModel):
    task_id: str
    party_task_id: str
    task_name: str
    component: str
    output_artifact_key: str
    output_index: Optional[int] = None

    def unique_key(self):
        key = f"{self.task_id}_{self.task_name}_{self.output_artifact_key}"
        if self.output_index is not None:
            key = f"{key}_index_{self.output_index}"
        return key


class Metadata(pydantic.BaseModel):
    metadata: dict = pydantic.Field(default_factory=dict)
    name: Optional[str] = None
    namespace: Optional[str] = None
    source: Optional[ArtifactSource] = None


class ModelOutputMetadata(pydantic.BaseModel):
    metadata: dict = pydantic.Field(default_factory=dict)
    name: Optional[str] = None
    namespace: Optional[str] = None
    source: Optional[ArtifactSource] = None
    model_overview: MLModelSpec = None

    class Config:
        extra = "forbid"


class DataOutputMetadata(pydantic.BaseModel):
    metadata: dict = pydantic.Field(default_factory=dict)
    name: Optional[str] = None
    namespace: Optional[str] = None
    source: Optional[ArtifactSource] = None
    data_overview: Optional[DataOverview] = None

    class Config:
        extra = "forbid"


class MetricOutputMetadata(pydantic.BaseModel):
    metadata: dict = pydantic.Field(default_factory=dict)
    name: Optional[str] = None
    namespace: Optional[str] = None
    source: Optional[ArtifactSource] = None

    class Config:
        extra = "forbid"


class ArtifactInputApplySpec(pydantic.BaseModel):
    uri: str
    metadata: Metadata
    type_name: Optional[str] = None


class ArtifactOutputApplySpec(pydantic.BaseModel):
    uri: str
    _is_template: Optional[bool] = None
    type_name: Optional[str] = None

    def is_template(self) -> bool:
        return "{index}" in self.uri

    def _check_is_template(self) -> bool:
        return "{index}" in self.uri

    @pydantic.validator("uri")
    def _check_uri(cls, v, values) -> str:
        if not _uri_regex.match(v):
            raise pydantic.ValidationError(f"`{v}` is not valid uri")
        return v


class IOArtifactMeta(pydantic.BaseModel):
    class InputMeta(pydantic.BaseModel):
        data: Dict[str, Union[List[Dict], Dict]]
        model: Dict[str, Union[List[Dict], Dict]]

    class OutputMeta(pydantic.BaseModel):
        data: Dict[str, Union[List[Dict], Dict]]
        model: Dict[str, Union[List[Dict], Dict]]
        metric: Dict[str, Union[List[Dict], Dict]]

    inputs: InputMeta
    outputs: OutputMeta
