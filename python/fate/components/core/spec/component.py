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

from fate.components.core.essential import Label, Role, Stage
from pydantic import BaseModel


class ParameterSpec(BaseModel):
    type: str
    default: Optional[Any]
    optional: bool
    description: str = ""
    type_meta: dict = {}


class ArtifactSpec(BaseModel):
    types: List[str]
    optional: bool
    stages: Optional[List[Stage]]
    roles: List[Role]
    description: str = ""
    is_multi: bool

    class Config:
        arbitrary_types_allowed = True

    def dict(self, *args, **kwargs):
        object_dict = super().dict(*args, **kwargs)
        object_dict["roles"] = [r.name for r in self.roles]
        object_dict["stages"] = [s.name for s in self.stages]
        return object_dict


class InputDefinitionsSpec(BaseModel):
    data: Dict[str, ArtifactSpec]
    model: Dict[str, ArtifactSpec]


class OutputDefinitionsSpec(BaseModel):
    data: Dict[str, ArtifactSpec]
    model: Dict[str, ArtifactSpec]
    metric: Dict[str, ArtifactSpec]


class ComponentSpec(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    name: str
    description: str
    provider: str
    version: str
    labels: List[Label]
    roles: List[Role]
    parameters: Dict[str, ParameterSpec]
    input_artifacts: InputDefinitionsSpec
    output_artifacts: OutputDefinitionsSpec

    def dict(self, *args, **kwargs):
        object_dict = super().dict(*args, **kwargs)
        object_dict["roles"] = [r.name for r in self.roles]
        object_dict["labels"] = [l.name for l in self.labels]
        return object_dict


class ComponentSpecV1(BaseModel):
    component: ComponentSpec
    schema_version: str = "v1"


class ArtifactTypeSpec(BaseModel):
    type_name: str
    uri_types: List[str]
    path_type: Literal["file", "directory", "distributed"]


class ComponentIOArtifactTypeSpec(BaseModel):
    name: str
    is_multi: bool
    optional: bool
    types: List[ArtifactTypeSpec]


class ComponentIOInputsArtifactsTypeSpec(BaseModel):
    data: List[ComponentIOArtifactTypeSpec]
    model: List[ComponentIOArtifactTypeSpec]


class ComponentIOOutputsArtifactsTypeSpec(BaseModel):
    data: List[ComponentIOArtifactTypeSpec]
    model: List[ComponentIOArtifactTypeSpec]
    metric: List[ComponentIOArtifactTypeSpec]


class ComponentIOArtifactsTypeSpec(BaseModel):
    inputs: ComponentIOInputsArtifactsTypeSpec
    outputs: ComponentIOOutputsArtifactsTypeSpec
