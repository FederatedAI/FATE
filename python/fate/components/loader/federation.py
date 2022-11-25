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

    # TODO: load from plugin
    raise ValueError(f"conf.federation={federation} not support")
