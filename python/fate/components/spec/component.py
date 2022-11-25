from typing import Any, Dict, List, Optional

from fate.components import T_LABEL, T_ROLE, T_STAGE
from pydantic import BaseModel


class ParameterSpec(BaseModel):
    type: str
    default: Any
    optional: bool
    description: str = ""


class ArtifactSpec(BaseModel):
    type: str
    optional: bool
    stages: Optional[List[T_STAGE]]
    roles: List[T_ROLE]


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
    labels: List[T_LABEL]
    roles: List[T_ROLE]
    input_definitions: InputDefinitionsSpec
    output_definitions: OutputDefinitionsSpec


class ComponentSpecV1(BaseModel):
    component: ComponentSpec
    schema_version: str = "v1"
