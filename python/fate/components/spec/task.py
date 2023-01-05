from typing import Any, Dict, List, Union

import pydantic

from .artifact import ArtifactSpec
from .computing import EggrollComputingSpec, SparkComputingSpec, StandaloneComputingSpec
from .device import CPUSpec, GPUSpec
from .federation import (
    EggrollFederationSpec,
    OSXFederationSpec,
    PulsarFederationSpec,
    RabbitMQFederationSpec,
    StandaloneFederationSpec,
)
from .logger import CustomLogger, FlowLogger, PipelineLogger
from .mlmd import CustomMLMDSpec, FlowMLMDSpec, NoopMLMDSpec, PipelineMLMDSpec
from .output import OutputPoolConf


class TaskConfigSpec(pydantic.BaseModel):
    class TaskInputsSpec(pydantic.BaseModel):
        parameters: Dict[str, Any] = {}
        artifacts: Dict[str, Union[ArtifactSpec, List[ArtifactSpec]]] = {}

    class TaskConfSpec(pydantic.BaseModel):
        device: Union[CPUSpec, GPUSpec]
        computing: Union[StandaloneComputingSpec, EggrollComputingSpec, SparkComputingSpec]
        federation: Union[
            StandaloneFederationSpec,
            EggrollFederationSpec,
            RabbitMQFederationSpec,
            PulsarFederationSpec,
            OSXFederationSpec,
        ]
        logger: Union[PipelineLogger, FlowLogger, CustomLogger]
        mlmd: Union[PipelineMLMDSpec, FlowMLMDSpec, NoopMLMDSpec, CustomMLMDSpec]
        output: OutputPoolConf

    task_id: str
    party_task_id: str
    component: str
    role: str
    party_id: str
    stage: str = "default"
    inputs: TaskInputsSpec = TaskInputsSpec(parameters={}, artifacts={})
    conf: TaskConfSpec
