from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from typing_extensions import Literal


class ParameterSpec(BaseModel):
    type: str
    default: Any
    optional: bool


class ArtifactSpec(BaseModel):
    type: str
    optional: bool
    stages: Optional[List[str]]
    roles: Optional[List[Literal["guest", "host", "arbiter"]]]


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
    roles: List[Literal["guest", "host", "arbiter"]]
    inputDefinitions: InputDefinitionsSpec
    outputDefinitions: OutputDefinitionsSpec


class ComponentSpecV1(BaseModel):
    component: ComponentSpec
    schemaVersion: str = "v1"
