from pydantic import BaseModel
from typing import Dict, Optional, Union, Any, List


class IOArtifact(BaseModel):
    name: str
    uri: str
    metadata: Optional[dict]


class InputSpec(BaseModel):
    parameters: Optional[Dict[str, Any]]
    artifacts: Optional[IOArtifact]


class TaskRuntimeInputSpec(BaseModel):
    parameters: Optional[Dict[str, str]]
    artifacts: Optional[Dict[str, IOArtifact]]


class TaskRuntimeOutputSpec(BaseModel):
    artifacts: Dict[str, IOArtifact]


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


class TaskScheduleSpec(BaseModel):
    execution_id: str
    component: str
    role: str
    stage: str
    party_id: Optional[Union[str, int]]
    inputs: Optional[TaskRuntimeInputSpec]
    outputs: Optional[Dict[str, IOArtifact]]
    conf: RuntimeConfSpec
