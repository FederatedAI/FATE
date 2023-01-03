from typing import Dict, List, Literal, Optional

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
        proxy_endpoint: str
        parties: FederationPartiesSpec

    type: Literal["eggroll"]
    metadata: MetadataSpec


class RabbitMQFederationSpec(pydantic.BaseModel):
    class MetadataSpec(pydantic.BaseModel):
        class RouteTable(pydantic.BaseModel):
            host: str
            port: int

        class RabbitMQConfig(pydantic.BaseModel):
            host: str
            port: int
            mng_port: int
            base_user: str
            base_password: str
            max_message_size: Optional[int] = None
            mode: str = "replication"

        federation_id: str
        parties: FederationPartiesSpec
        route_table: Dict[str, RouteTable]
        rabbitmq_config: RabbitMQConfig
        rabbitmq_run: dict = {}
        connection: dict = {}

    type: Literal["rabbitmq"]
    metadata: MetadataSpec


class PulsarFederationSpec(pydantic.BaseModel):
    class MetadataSpec(pydantic.BaseModel):
        class RouteTable(pydantic.BaseModel):
            host: str
            port: int

        class PulsarConfig(pydantic.BaseModel):
            host: str
            port: int
            mng_port: int
            base_user: Optional[str] = None
            base_password: Optional[str] = None
            max_message_size: Optional[int] = None
            mode: str = "replication"
            topic_ttl: Optional[int] = None
            cluster: Optional[str] = None
            tenant: Optional[str] = None

        federation_id: str
        parties: FederationPartiesSpec
        route_table: Dict[str, RouteTable]
        pulsar_config: PulsarConfig
        pulsar_run: dict = {}
        connection: dict = {}

    type: Literal["pulsar"]
    metadata: MetadataSpec


class CustomFederationSpec(pydantic.BaseModel):
    type: Literal["custom"]
    metadata: dict
