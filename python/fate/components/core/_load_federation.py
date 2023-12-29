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
    from fate.arch.federation import FederationBuilder, FederationMode
    from fate.components.core.spec.federation import (
        OSXFederationSpec,
        PulsarFederationSpec,
        RabbitMQFederationSpec,
        RollSiteFederationSpec,
        StandaloneFederationSpec,
    )

    builder = FederationBuilder(
        federation_session_id=federation.metadata.federation_id,
        party=federation.metadata.parties.local.tuple(),
        parties=[p.tuple() for p in federation.metadata.parties.parties],
    )

    if isinstance(federation, StandaloneFederationSpec):
        return builder.build_standalone(
            computing_session=computing,
        )

    if isinstance(federation, (OSXFederationSpec, RollSiteFederationSpec)):
        if isinstance(federation, OSXFederationSpec):
            mode = FederationMode.from_str(federation.metadata.osx_config.mode)
            host = federation.metadata.osx_config.host
            port = federation.metadata.osx_config.port
            options = dict(max_message_size=federation.metadata.osx_config.max_message_size)
        else:
            mode = FederationMode.STREAM
            host = federation.metadata.rollsite_config.host
            port = federation.metadata.rollsite_config.port
            options = {}
        return builder.build_osx(
            computing_session=computing,
            host=host,
            port=port,
            mode=mode,
            options=options,
        )
    if isinstance(federation, RabbitMQFederationSpec):
        return builder.build_rabbitmq(
            computing_session=computing,
            host=federation.metadata.rabbitmq_config.host,
            port=federation.metadata.rabbitmq_config.port,
            options=dict(
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
            ),
        )

    if isinstance(federation, PulsarFederationSpec):
        route_table = {}
        for k, v in federation.metadata.route_table.route.items():
            route_table.update({k: v.dict()})
        if (default := federation.metadata.route_table.default) is not None:
            route_table.update({"default": default.dict()})
        return builder.build_pulsar(
            computing_session=computing,
            host=federation.metadata.pulsar_config.host,
            port=federation.metadata.pulsar_config.port,
            options=dict(
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
            ),
        )

    # TODO: load from plugin
    raise ValueError(f"conf.federation={federation} not support")
