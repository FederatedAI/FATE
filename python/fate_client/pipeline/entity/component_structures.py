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
from typing import Optional, Dict, List, Union, Any
from pathlib import Path
from pydantic import BaseModel
from ..utils.file_utils import load_yaml_file

TypeSpecType = Union[str, Dict, List]


class ParameterSpec(BaseModel):
    type: str
    default: Any
    optional: bool


class ArtifactSpec(BaseModel):
    type: str
    optional: bool
    stages: Optional[List[str]]
    roles: Optional[List[str]]


class InputDefinitionsSpec(BaseModel):
    parameters: Dict[str, ParameterSpec]
    artifacts: Dict[str, ArtifactSpec]


class OutputDefinitionsSpec(BaseModel):
    artifacts: Dict[str, ArtifactSpec]


class ComponentSpec(BaseModel):
    name: str
    description: str
    provider: str
    version: str
    labels: List[str] = ["trainable"]
    roles: List[str]
    input_definitions: InputDefinitionsSpec
    output_definitions: OutputDefinitionsSpec


class RuntimeOutputChannelSpec(BaseModel):
    producer_task: str
    output_artifact_key: str


class RuntimeInputDefinition(BaseModel):
    parameters: Optional[Dict[str, Any]]
    artifacts: Optional[Dict[str, Dict[str, RuntimeOutputChannelSpec]]]


def load_component_spec(yaml_define_path: str):
    yaml_define_path = Path(__file__).parent.parent.joinpath(yaml_define_path).resolve()
    component_spec_dict = load_yaml_file(str(yaml_define_path))["component"]
    parameters = dict()
    input_artifacts = dict()
    output_artifacts = dict()
    if "input_definitions" in component_spec_dict:
        input_definition_spec = component_spec_dict["input_definitions"]
        if "parameters" in input_definition_spec:
            for key, value in input_definition_spec["parameters"].items():
                parameters[key] = ParameterSpec(**value)

        if "artifacts" in input_definition_spec:
            for key, value in input_definition_spec["artifacts"].items():
                input_artifacts[key] = ArtifactSpec(**value)

    if "output_definitions" in component_spec_dict:
        output_definition_spec = component_spec_dict["output_definitions"]
        for key, value in output_definition_spec["artifacts"].items():
            output_artifacts[key] = ArtifactSpec(**value)

    input_definitions = InputDefinitionsSpec(parameters=parameters,
                                             artifacts=input_artifacts)

    output_definitions = OutputDefinitionsSpec(artifacts=output_artifacts)

    return ComponentSpec(
        name=component_spec_dict["name"],
        description=component_spec_dict["description"],
        provider=component_spec_dict["provider"],
        version=component_spec_dict["version"],
        labels=component_spec_dict["labels"],
        roles=component_spec_dict["roles"],
        input_definitions=input_definitions,
        output_definitions=output_definitions
    )
