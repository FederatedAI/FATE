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
import os
from typing import Any, Dict, List, Optional, Union

import pydantic

from .artifact import ArtifactInputApplySpec, ArtifactOutputApplySpec
from .computing import EggrollComputingSpec, SparkComputingSpec, StandaloneComputingSpec
from .device import CPUSpec, GPUSpec
from .federation import (
    OSXFederationSpec,
    PulsarFederationSpec,
    RabbitMQFederationSpec,
    RollSiteFederationSpec,
    StandaloneFederationSpec,
)
from .logger import LoggerConfig


class TaskConfigSpec(pydantic.BaseModel):
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
        logger: LoggerConfig
        task_final_meta_path: pydantic.FilePath = pydantic.Field(default_factory=lambda: os.path.abspath(os.getcwd()))

    task_id: str
    party_task_id: str
    task_name: str
    component: str
    role: str
    party_id: str
    stage: str = "default"
    parameters: Dict[str, Any] = {}
    input_artifacts: Dict[str, Optional[Union[List[ArtifactInputApplySpec], ArtifactInputApplySpec]]] = {}
    output_artifacts: Dict[str, Optional[ArtifactOutputApplySpec]] = {}
    conf: TaskConfSpec


class TaskCleanupConfigSpec(pydantic.BaseModel):
    computing: Union[StandaloneComputingSpec, EggrollComputingSpec, SparkComputingSpec]
    federation: Union[
        StandaloneFederationSpec,
        RollSiteFederationSpec,
        RabbitMQFederationSpec,
        PulsarFederationSpec,
        OSXFederationSpec,
    ]
