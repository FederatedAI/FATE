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
    created_time: datetime
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
