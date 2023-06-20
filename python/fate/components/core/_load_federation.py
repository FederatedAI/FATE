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
def load_federation(federation, computing):
    from fate.components.core.spec.federation import (
        OSXFederationSpec,
        PulsarFederationSpec,
        RabbitMQFederationSpec,
        RollSiteFederationSpec,
        StandaloneFederationSpec,
    )

    if isinstance(federation, StandaloneFederationSpec):
        from fate.arch.federation.standalone import StandaloneFederation

        return StandaloneFederation(
            computing,
            federation.metadata.federation_id,
            federation.metadata.parties.local.tuple(),
            [p.tuple() for p in federation.metadata.parties.parties],
        )

    if isinstance(federation, RollSiteFederationSpec):
        from fate.arch.computing.eggroll import CSession
        from fate.arch.federation.eggroll import EggrollFederation

        if not isinstance(computing, CSession):
            raise RuntimeError(f"Eggroll federation type requires Eggroll computing type, `{type(computing)}` found")

        return EggrollFederation(
            rp_ctx=computing.get_rpc(),
            rs_session_id=federation.metadata.federation_id,
            party=federation.metadata.parties.local.tuple(),
            parties=[p.tuple() for p in federation.metadata.parties.parties],
            proxy_endpoint=f"{federation.metadata.rollsite_config.host}:{federation.metadata.rollsite_config.port}",
        )

    if isinstance(federation, RabbitMQFederationSpec):
        from fate.arch.federation.rabbitmq import RabbitmqFederation

        return RabbitmqFederation.from_conf(
            federation_session_id=federation.metadata.federation_id,
            computing_session=computing,
            party=federation.metadata.parties.local.tuple(),
            parties=[p.tuple() for p in federation.metadata.parties.parties],
            route_table={k: v.dict() for k, v in federation.metadata.route_table.items()},
            host=federation.metadata.rabbitmq_config.host,
            port=federation.metadata.rabbitmq_config.port,
            mng_port=federation.metadata.rabbitmq_config.mng_port,
            base_user=federation.metadata.rabbitmq_config.user,
            base_password=federation.metadata.rabbitmq_config.password,
            mode=federation.metadata.rabbitmq_config.mode,
            max_message_size=federation.metadata.rabbitmq_config.max_message_size,
            rabbitmq_run=federation.metadata.rabbitmq_run,
            connection=federation.metadata.connection,
        )

    if isinstance(federation, PulsarFederationSpec):
        from fate.arch.federation.pulsar import PulsarFederation

        route_table = {}
        for k, v in federation.metadata.route_table.route.items():
            route_table.update({k: v.dict()})
        if (default := federation.metadata.route_table.default) is not None:
            route_table.update({"default": default.dict()})
        return PulsarFederation.from_conf(
            federation_session_id=federation.metadata.federation_id,
            computing_session=computing,
            party=federation.metadata.parties.local.tuple(),
            parties=[p.tuple() for p in federation.metadata.parties.parties],
            route_table=route_table,
            mode=federation.metadata.pulsar_config.mode,
            host=federation.metadata.pulsar_config.host,
            port=federation.metadata.pulsar_config.port,
            mng_port=federation.metadata.pulsar_config.mng_port,
            base_user=federation.metadata.pulsar_config.user,
            base_password=federation.metadata.pulsar_config.password,
            max_message_size=federation.metadata.pulsar_config.max_message_size,
            topic_ttl=federation.metadata.pulsar_config.topic_ttl,
            cluster=federation.metadata.pulsar_config.cluster,
            tenant=federation.metadata.pulsar_config.tenant,
            pulsar_run=federation.metadata.pulsar_run,
            connection=federation.metadata.connection,
        )

    if isinstance(federation, OSXFederationSpec):
        from fate.arch.federation.osx import OSXFederation

        return OSXFederation.from_conf(
            federation_session_id=federation.metadata.federation_id,
            computing_session=computing,
            party=federation.metadata.parties.local.tuple(),
            parties=[p.tuple() for p in federation.metadata.parties.parties],
            host=federation.metadata.osx_config.host,
            port=federation.metadata.osx_config.port,
            max_message_size=federation.metadata.osx_config.max_message_size,
        )
    # TODO: load from plugin
    raise ValueError(f"conf.federation={federation} not support")
