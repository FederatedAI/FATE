from typing import Any, Dict, List, Optional, Union

import pydantic

from .artifact import ArtifactSpec
from .computing import EggrollComputingSpec, SparkComputingSpec, StandaloneComputingSpec
from .device import CPUSpec, GPUSpec
from .federation import (
    EggrollFederationSpec,
    RabbitMQFederationSpec,
    StandaloneFederationSpec,
)
from .logger import CustomLogger, FlowLogger, PipelineLogger
from .mlmd import CustomMLMDSpec, FlowMLMDSpec, PipelineMLMDSpec
from .output import OutputPoolConf


class OtherTaskOutputTaskInputArtifactSpec(pydantic.BaseModel):
    class OtherTaskOutputArtifact(pydantic.BaseModel):
        output_artifact_key: str
        producer_task: str

    name: str
    metadata: Optional[dict] = None
    task_output_artifact: OtherTaskOutputArtifact


class TaskConfigSpec(pydantic.BaseModel):
    class TaskInputsSpec(pydantic.BaseModel):
        parameters: Dict[str, Any] = {}
        artifacts: Dict[str, Union[ArtifactSpec, List[ArtifactSpec]]] = {}

    class TaskConfSpec(pydantic.BaseModel):
        device: Union[CPUSpec, GPUSpec]
        computing: Union[StandaloneComputingSpec, EggrollComputingSpec, SparkComputingSpec]
        federation: Union[StandaloneFederationSpec, EggrollFederationSpec, RabbitMQFederationSpec]
        logger: Union[PipelineLogger, FlowLogger, CustomLogger]
        mlmd: Union[PipelineMLMDSpec, FlowMLMDSpec, CustomMLMDSpec]
        output: OutputPoolConf

    execution_id: str
    component: str
    role: str
    stage: str = "default"
    inputs: TaskInputsSpec = TaskInputsSpec(parameters={}, artifacts={})
    conf: TaskConfSpec
