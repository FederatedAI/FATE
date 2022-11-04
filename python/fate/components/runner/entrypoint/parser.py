from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Union

import pydantic
from fate.arch.unify import device

from .logger import CustomLogger, FlowLogger, PipelineLogger


class TaskExecuteStatus(Enum):
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"


class FATECpnExtra(pydantic.BaseModel):
    class DistributedComputingBackend(pydantic.BaseModel):
        engine: str
        computing_id: str

    class FederationBackend(pydantic.BaseModel):
        class Parties(pydantic.BaseModel):
            local: Tuple[Literal["guest", "host", "arbiter"], str]
            parties: List[Tuple[Literal["guest", "host", "arbiter"], str]]

        engine: str
        federation_id: str
        parties: Parties

    device: device
    distributed_computing_backend: DistributedComputingBackend
    federation_backend: FederationBackend
    logger: Union[PipelineLogger, FlowLogger, CustomLogger]

    @pydantic.validator("device", pre=True)
    def _device_validate(cls, v):
        if not isinstance(v, str):
            raise ValueError("must be str")
        for dev in device:
            if dev.name == v.strip().upper():
                return dev
        raise ValueError(f"should be one of {[dev.name for dev in device]}")


class FATEComponentTaskConfig(pydantic.BaseModel):
    task_id: str
    cpn: str
    role: str
    params: Dict
    data_inputs: List[str]
    data_outputs: List[str]
    model_inputs: List[str]
    model_outputs: List[str]
    status_output: str
    extra: FATECpnExtra
    metrics_output: Optional[str] = None
    status_input: Optional[str] = None
