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
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class ParameterSpec(BaseModel):
    type: str
    default: Any
    optional: bool
    description: str = ""
    type_meta: dict = {}


class ArtifactSpec(BaseModel):
    type: str
    optional: bool
    stages: Optional[Literal["train", "predict", "default", "cross_validation"]]
    roles: Literal["guest", "host", "arbiter", "local"]
    description: str = ""
    is_multi: bool


class InputDefinitionsSpec(BaseModel):
    data: Dict[str, ArtifactSpec]
    model: Dict[str, ArtifactSpec]


class OutputDefinitionsSpec(BaseModel):
    data: Dict[str, ArtifactSpec]
    model: Dict[str, ArtifactSpec]
    metric: Dict[str, ArtifactSpec]


class ComponentSpec(BaseModel):
    name: str
    description: str
    provider: str
    version: str
    labels: List[Literal["trainable"]]
    roles: Literal["guest", "host", "arbiter", "local"]
    parameters: Dict[str, ParameterSpec]
    input_artifacts: InputDefinitionsSpec
    output_artifacts: OutputDefinitionsSpec


class ComponentSpecV1(BaseModel):
    component: ComponentSpec
    schema_version: str = "v1"
