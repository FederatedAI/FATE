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
from pydantic import BaseModel
from typing import Dict, Optional, Union, Any, List


class InputArtifact(BaseModel):
    name: str
    uri: str
    metadata: Optional[dict]


class OutputArtifact(BaseModel):
    type: str
    metadata: Optional[dict]


class InputSpec(BaseModel):
    parameters: Optional[Dict[str, Any]]
    artifacts: Optional[InputArtifact]


class TaskRuntimeInputSpec(BaseModel):
    parameters: Optional[Dict[str, str]]
    artifacts: Optional[Dict[str, InputArtifact]]


class TaskRuntimeOutputSpec(BaseModel):
    artifacts: Dict[str, OutputArtifact]


class MLMDSpec(BaseModel):
    type: str
    metadata: Dict[str, Any]


class LOGGERSpec(BaseModel):
    type: str
    metadata: Dict[str, Any]


class ComputingEngineMetadata(BaseModel):
    computing_id: str


class ComputingEngineSpec(BaseModel):
    type: str
    metadata: ComputingEngineMetadata


class DeviceSpec(BaseModel):
    type: str


class FederationPartySpec(BaseModel):
    local: Dict[str, str]
    parties: List[Dict[str, str]]


class FederationEngineMetadata(BaseModel):
    federation_id: str
    parties: FederationPartySpec


class FederationEngineSpec(BaseModel):
    type: str
    metadata: FederationEngineMetadata


class RuntimeConfSpec(BaseModel):
    mlmd: MLMDSpec
    logger: LOGGERSpec
    device: DeviceSpec
    computing: ComputingEngineSpec
    federation: FederationEngineSpec
    output: Dict[str, OutputArtifact]


class TaskScheduleSpec(BaseModel):
    task_id: str
    party_task_id: str
    component: str
    role: str
    stage: str
    party_id: Optional[str]
    inputs: Optional[TaskRuntimeInputSpec]
    conf: RuntimeConfSpec
