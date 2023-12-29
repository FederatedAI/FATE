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
from datetime import datetime
from typing import List

import pydantic


class MLModelComponentSpec(pydantic.BaseModel):
    name: str
    provider: str
    version: str
    metadata: dict


class MLModelPartiesSpec(pydantic.BaseModel):
    guest: List[str]
    host: List[str]
    arbiter: List[str]


class MLModelFederatedSpec(pydantic.BaseModel):
    task_id: str
    parties: MLModelPartiesSpec
    component: MLModelComponentSpec


class MLModelModelSpec(pydantic.BaseModel):
    name: str
    created_time: str
    file_format: str
    metadata: dict


class MLModelPartySpec(pydantic.BaseModel):
    party_task_id: str
    role: str
    partyid: str
    models: List[MLModelModelSpec]


class MLModelSpec(pydantic.BaseModel):
    federated: MLModelFederatedSpec
    party: MLModelPartySpec
