from pydantic import BaseModel
from typing import Optional, Literal, List, Union, Dict, Any, TypeVar


class PartySpec(BaseModel):
    role: Union[Literal["guest", "host", "arbiter"]]
    party_id: List[Union[str, int]]


class RuntimeTaskOutputChannelSpec(BaseModel):
    producer_task: str
    output_artifact_key: str


class ModelWarehouseChannelSpec(BaseModel):
    model_id: Optional[str]
    model_version: Optional[str]
    producer_task: str
    output_artifact_key: str


InputChannelSpec = TypeVar("InputChannelSpec", RuntimeTaskOutputChannelSpec, ModelWarehouseChannelSpec)


class RuntimeInputDefinition(BaseModel):
    parameters: Optional[Dict[str, Any]]
    artifacts: Optional[Dict[str, Dict[str, Union[InputChannelSpec, List[InputChannelSpec]]]]]


class TaskSpec(BaseModel):
    component_ref: str
    dependent_tasks: Optional[List[str]]
    inputs: Optional[RuntimeInputDefinition]
    parties: Optional[List[PartySpec]]
    conf: Optional[Dict[Any, Any]]
    stage: Optional[Union[Literal["train", "predict", "default"]]]


class PartyTaskRefSpec(BaseModel):
    inputs: RuntimeInputDefinition
    conf: Optional[Dict]


class PartyTaskSpec(BaseModel):
    parties: Optional[List[PartySpec]]
    tasks: Dict[str, PartyTaskRefSpec]
    conf: Optional[dict]


class TaskConfSpec(BaseModel):
    task_cores: int
    engine: Dict[str, Any]


class JobConfSpec(BaseModel):
    inherit: Optional[Dict[str, Any]]
    task_parallelism: Optional[int]
    federated_status_collect_type: Optional[str]
    auto_retries: Optional[int]
    model_id: Optional[str]
    model_version: Optional[str]
    task: Optional[TaskConfSpec]


class DAGSpec(BaseModel):
    scheduler_party_id: Union[str, int]
    parties: List[PartySpec]
    conf: Optional[JobConfSpec]
    stage: Optional[Union[Literal["train", "predict", "default"]]]
    tasks: Dict[str, TaskSpec]
    party_tasks: Optional[Dict[str, PartyTaskSpec]]


class DAGSchema(BaseModel):
    dag: DAGSpec
    schema_version: str
