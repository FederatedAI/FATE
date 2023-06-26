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
from typing import Dict, List, Literal, Optional

import pydantic


class PartySpec(pydantic.BaseModel):
    role: Literal["guest", "host", "arbiter", "local"]
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


class RollSiteFederationSpec(pydantic.BaseModel):
    class MetadataSpec(pydantic.BaseModel):
        class RollSiteConfig(pydantic.BaseModel):
            host: str
            port: int

        federation_id: str
        parties: FederationPartiesSpec
        rollsite_config: RollSiteConfig

    type: Literal["rollsite"]
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
            user: str
            password: str
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
            class Route(pydantic.BaseModel):
                host: str
                port: int
                sslPort: int
                proxy: str = ""

            class Default(pydantic.BaseModel):
                domain: str
                brokerPort: int
                brokerSslPort: int
                proxy: str = ""

            route: Dict[str, Route]
            default: Optional[Default] = None

        class PulsarConfig(pydantic.BaseModel):
            host: str
            port: int
            mng_port: int
            user: Optional[str] = None
            password: Optional[str] = None
            max_message_size: Optional[int] = None
            mode: str = "replication"
            topic_ttl: Optional[int] = None
            cluster: Optional[str] = None
            tenant: Optional[str] = None

        federation_id: str
        parties: FederationPartiesSpec
        route_table: RouteTable
        pulsar_config: PulsarConfig
        pulsar_run: dict = {}
        connection: dict = {}

    type: Literal["pulsar"]
    metadata: MetadataSpec


class OSXFederationSpec(pydantic.BaseModel):
    class MetadataSpec(pydantic.BaseModel):
        class OSXConfig(pydantic.BaseModel):
            host: str
            port: int
            max_message_size: Optional[int] = None

        federation_id: str
        parties: FederationPartiesSpec
        osx_config: OSXConfig

    type: Literal["osx"]
    metadata: MetadataSpec


class CustomFederationSpec(pydantic.BaseModel):
    type: Literal["custom"]
    metadata: dict
