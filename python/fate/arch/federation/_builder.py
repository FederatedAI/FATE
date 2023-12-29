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

import typing
from enum import Enum

from fate.arch.federation.api import PartyMeta
from fate.arch.config import cfg


class FederationEngine(Enum):
    STANDALONE = "standalone"
    OSX = "osx"
    RABBITMQ = "rabbitmq"
    PULSAR = "pulsar"

    @classmethod
    def from_str(cls, s: str):
        for t in cls:
            if t.value == s:
                return t
        raise ValueError(f"{s} not in {cls}")

    def __str__(self):
        return self.value


class FederationMode(Enum):
    STREAM = "stream"
    MESSAGE_QUEUE = "message_queue"

    @classmethod
    def from_str(cls, s: str):
        if isinstance(s, cls):
            return s
        for t in cls:
            if t.value == s:
                return t
        raise ValueError(f"{s} not in {cls}")


class FederationBuilder:
    def __init__(
        self,
        federation_session_id: str,
        party: PartyMeta,
        parties: typing.List[PartyMeta],
    ):
        self._federation_id = federation_session_id
        self._party = party
        self._parties = parties

    def build(self, computing_session, t: FederationEngine, conf: dict):
        if t == FederationEngine.STANDALONE:
            return self.build_standalone(computing_session)
        elif t == FederationEngine.OSX:
            host = cfg.get_option(conf, "federation.osx.host")
            port = cfg.get_option(conf, "federation.osx.port")
            mode = FederationMode.from_str(cfg.get_option(conf, "federation.osx.mode", FederationMode.MESSAGE_QUEUE))
            return self.build_osx(computing_session, host=host, port=port, mode=mode)
        elif t == FederationEngine.RABBITMQ:
            host = cfg.get_option(conf, "federation.rabbitmq.host")
            port = cfg.get_option(conf, "federation.rabbitmq.port")
            options = cfg.get_option(conf, "federation.rabbitmq")
            return self.build_rabbitmq(computing_session, host=host, port=port, options=options)
        elif t == FederationEngine.PULSAR:
            host = cfg.get_option(conf, "federation.pulsar.host")
            port = cfg.get_option(conf, "federation.pulsar.port")
            options = cfg.get_option(conf, "federation.pulsar")
            return self.build_pulsar(computing_session, host=host, port=port, options=options)
        else:
            raise ValueError(f"{t} not in {FederationEngine}")

    def build_standalone(self, computing_session):
        from fate.arch.federation.backends.standalone import StandaloneFederation

        return StandaloneFederation(
            standalone_session=computing_session,
            federation_session_id=self._federation_id,
            party=self._party,
            parties=self._parties,
        )

    def build_osx(
        self, computing_session, host: str, port: int, mode=FederationMode.MESSAGE_QUEUE, options: dict = None
    ):
        if options is None:
            options = {}
        if mode == FederationMode.MESSAGE_QUEUE:
            from fate.arch.federation.backends.osx import OSXFederation

            return OSXFederation.from_conf(
                federation_session_id=self._federation_id,
                computing_session=computing_session,
                party=self._party,
                parties=self._parties,
                host=host,
                port=port,
                max_message_size=options.get("max_message_size"),
            )
        else:
            from fate.arch.computing.backends.eggroll import CSession
            from fate.arch.federation.backends.eggroll import EggrollFederation

            if not isinstance(computing_session, CSession):
                raise RuntimeError(
                    f"Eggroll federation type requires Eggroll computing type, `{type(computing_session)}` found"
                )

            return EggrollFederation(
                computing_session=computing_session,
                federation_session_id=self._federation_id,
                party=self._party,
                parties=self._parties,
                proxy_endpoint=f"{host}:{port}",
            )

    def build_rabbitmq(self, computing_session, host: str, port: int, options: dict):
        from fate.arch.federation.backends.rabbitmq import RabbitmqFederation

        return RabbitmqFederation.from_conf(
            federation_session_id=self._federation_id,
            computing_session=computing_session,
            party=self._party,
            parties=self._parties,
            host=host,
            port=port,
            route_table=options["route_table"],
            mng_port=options["mng_port"],
            base_user=options["base_user"],
            base_password=options["base_password"],
            mode=options["mode"],
            max_message_size=options["max_message_size"],
            rabbitmq_run=options["rabbitmq_run"],
            connection=options["connection"],
        )

    def build_pulsar(self, computing_session, host: str, port: int, options: dict):
        from fate.arch.federation.backends.pulsar import PulsarFederation

        return PulsarFederation.from_conf(
            federation_session_id=self._federation_id,
            computing_session=computing_session,
            party=self._party,
            parties=self._parties,
            host=host,
            port=port,
            route_table=options["route_table"],
            mode=options["mode"],
            mng_port=options["mng_port"],
            base_user=options["base_user"],
            base_password=options["base_password"],
            max_message_size=options["max_message_size"],
            topic_ttl=options["topic_ttl"],
            cluster=options["cluster"],
            tenant=options["tenant"],
            pulsar_run=options["pulsar_run"],
            connection=options["connection"],
        )
