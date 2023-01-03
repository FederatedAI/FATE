def load_federation(federation, computing):
    from fate.components.spec.federation import (
        EggrollFederationSpec,
        RabbitMQFederationSpec,
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

    if isinstance(federation, EggrollFederationSpec):
        from fate.arch.computing.eggroll import CSession
        from fate.arch.federation.eggroll import EggrollFederation

        if not isinstance(computing, CSession):
            raise RuntimeError(f"Eggroll federation type requires Eggroll computing type, `{type(computing)}` found")

        return EggrollFederation(
            rp_ctx=computing.get_rpc(),
            rs_session_id=federation.metadata.federation_id,
            party=federation.metadata.parties.local.tuple(),
            parties=[p.tuple() for p in federation.metadata.parties.parties],
            proxy_endpoint=federation.metadata.proxy_endpoint,
        )

    if isinstance(federation, RabbitMQFederationSpec):
        from fate.arch.federation.rabbitmq import RabbitmqFederation

        return RabbitmqFederation.from_conf(
            federation_session_id=federation.metadata.federation_id,
            party=federation.metadata.parties.local.tuple(),
            parties=[p.tuple() for p in federation.metadata.parties.parties],
            route_table={k: v.dict() for k, v in federation.metadata.route_table.items()},
            host=federation.metadata.rabbitmq_config.host,
            port=federation.metadata.rabbitmq_config.port,
            mng_port=federation.metadata.rabbitmq_config.mng_port,
            base_user=federation.metadata.rabbitmq_config.base_user,
            base_password=federation.metadata.rabbitmq_config.base_password,
            mode=federation.metadata.rabbitmq_config.mode,
            max_message_size=federation.metadata.rabbitmq_config.max_message_size,
            rabbitmq_run=federation.metadata.rabbitmq_run,
            connection=federation.metadata.connection,
        )
    # TODO: load from plugin
    raise ValueError(f"conf.federation={federation} not support")
