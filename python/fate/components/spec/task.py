from typing import Any, Dict, List, Literal, Optional, Union

import pydantic
from fate.components.spec.computing import (
    EggrollComputingSpec,
    SparkComputingSpec,
    StandaloneComputingSpec,
)
from fate.components.spec.federation import (
    EggrollFederationSpec,
    RabbitMQFederationSpec,
    StandaloneFederationSpec,
)
from fate.components.spec.mlmd import CustomMLMDSpec, FlowMLMDSpec, PipelineMLMDSpec

from .logger import CustomLogger, FlowLogger, PipelineLogger


class TaskConfSpec(pydantic.BaseModel):
    device: Literal["CPU", "GPU"]
    computing: Union[StandaloneComputingSpec, EggrollComputingSpec, SparkComputingSpec]
    federation: Union[StandaloneFederationSpec, EggrollFederationSpec, RabbitMQFederationSpec]
    logger: Union[PipelineLogger, FlowLogger, CustomLogger]
    mlmd: Union[PipelineMLMDSpec, FlowMLMDSpec, CustomMLMDSpec]

    def get_device(self):
        from fate.arch.unify import device

        for dev in device:
            if dev.name == self.device.strip().upper():
                return dev
        raise ValueError(f"should be one of {[dev.name for dev in device]}")


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
