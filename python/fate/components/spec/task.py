from typing import Any, Dict, List, Optional, Union

import pydantic

from .computing import EggrollComputingSpec, SparkComputingSpec, StandaloneComputingSpec
from .device import CPUSpec, GPUSpec
from .federation import (
    EggrollFederationSpec,
    RabbitMQFederationSpec,
    StandaloneFederationSpec,
)
from .logger import CustomLogger, FlowLogger, PipelineLogger
from .mlmd import CustomMLMDSpec, FlowMLMDSpec, PipelineMLMDSpec


class TaskConfSpec(pydantic.BaseModel):
    device: Union[CPUSpec, GPUSpec]
    computing: Union[StandaloneComputingSpec, EggrollComputingSpec, SparkComputingSpec]
    federation: Union[StandaloneFederationSpec, EggrollFederationSpec, RabbitMQFederationSpec]
    logger: Union[PipelineLogger, FlowLogger, CustomLogger]
    mlmd: Union[PipelineMLMDSpec, FlowMLMDSpec, CustomMLMDSpec]


class ArtifactSpec(pydantic.BaseModel):
    name: str
    uri: str
    metadata: Optional[dict] = None


class TaskInputsSpec(pydantic.BaseModel):
    parameters: Dict[str, Any] = {}
    artifacts: Dict[str, Union[ArtifactSpec, List[ArtifactSpec]]] = {}


class TaskOutputsSpec(pydantic.BaseModel):
    artifacts: Dict[str, Union[ArtifactSpec, List[ArtifactSpec]]] = {}


class TaskConfigSpec(pydantic.BaseModel):
    execution_id: str
    component: str
    role: str
    stage: str = "default"
    inputs: TaskInputsSpec = TaskInputsSpec(parameters={}, artifacts={})
    outputs: TaskOutputsSpec = TaskOutputsSpec(artifacts={})
    conf: TaskConfSpec
