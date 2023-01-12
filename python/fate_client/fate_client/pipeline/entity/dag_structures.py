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
from typing import Optional, Literal, List, Union, Dict, Any, TypeVar


class PartySpec(BaseModel):
    role: Union[Literal["guest", "host", "arbiter"]]
    party_id: List[str]


class OutputChannelSpec(BaseModel):
    producer_task: str
    output_artifact_key: str
    model_id: Optional[str]
    model_version: Optional[int]
    roles: Optional[List[Literal["guest", "host", "arbiter"]]]


class RuntimeTaskOutputChannelSpec(BaseModel):
    producer_task: str
    output_artifact_key: str
    roles: Optional[List[Literal["guest", "host", "arbiter"]]]


class ModelWarehouseChannelSpec(BaseModel):
    model_id: Optional[str]
    model_version: Optional[int]
    producer_task: str
    output_artifact_key: str
    roles: Optional[List[Literal["guest", "host", "arbiter"]]]


InputChannelSpec = TypeVar("InputChannelSpec",
                           OutputChannelSpec,
                           RuntimeTaskOutputChannelSpec,
                           ModelWarehouseChannelSpec)


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
    model_version: Optional[int]
    task: Optional[TaskConfSpec]


class DAGSpec(BaseModel):
    parties: List[PartySpec]
    conf: Optional[JobConfSpec]
    stage: Optional[Union[Literal["train", "predict", "default"]]]
    tasks: Dict[str, TaskSpec]
    party_tasks: Optional[Dict[str, PartyTaskSpec]]


class DAGSchema(BaseModel):
    dag: DAGSpec
    schema_version: str
