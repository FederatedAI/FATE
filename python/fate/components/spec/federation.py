from typing import List, Literal

import pydantic


class PartySpec(pydantic.BaseModel):
    role: Literal["guest", "host", "arbiter"]
    partyid: str

    def tuple(self):
        return (self.role, self.partyid)


class FederationPartiesSpec(pydantic.BaseModel):
    local: PartySpec
    parties: List[PartySpec]


class StandaloneFederationSpec(pydantic.BaseModel):
    class MetadataSpec(pydantic.BaseModel):
        federation_id: str
        parties: FederationPartiesSpec

    type: Literal["standalone"]
    metadata: MetadataSpec


class EggrollFederationSpec(pydantic.BaseModel):
    class MetadataSpec(pydantic.BaseModel):
        federation_id: str
        parties: FederationPartiesSpec

    type: Literal["eggroll"]
    metadata: MetadataSpec


class RabbitMQFederationSpec(pydantic.BaseModel):
    class MetadataSpec(pydantic.BaseModel):
        federation_id: str
        parties: FederationPartiesSpec

    type: Literal["rabbitmq"]
    metadata: MetadataSpec


class CustomFederationSpec(pydantic.BaseModel):
    type: Literal["custom"]
    metadata: dict
