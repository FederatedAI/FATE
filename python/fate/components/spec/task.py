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
from typing import Any, Dict, List, Union

import pydantic

from .artifact import ArtifactSpec
from .computing import EggrollComputingSpec, SparkComputingSpec, StandaloneComputingSpec
from .device import CPUSpec, GPUSpec
from .federation import (
    OSXFederationSpec,
    PulsarFederationSpec,
    RabbitMQFederationSpec,
    RollSiteFederationSpec,
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
            RollSiteFederationSpec,
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
