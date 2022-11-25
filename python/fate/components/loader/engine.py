from fate.components.spec.task import TaskComputingSpec, TaskFederationSpec


def load_computing(computing: TaskComputingSpec):
    if (engine := computing.engine) == "standalone":
        from fate.arch.computing.standalone import CSession

        return CSession(computing.computing_id)
    elif engine == "eggroll":
        from fate.arch.computing.eggroll import CSession

        return CSession(computing.computing_id)
    elif engine == "spark":
        from fate.arch.computing.spark import CSession

        return CSession(computing.computing_id)

    else:
        raise ValueError(f"conf.computing.engine={engine} not support")


def load_federation(federation: TaskFederationSpec, computing):
    if (engine := federation.engine) == "standalone":
        from fate.arch.federation.standalone import StandaloneFederation

        return StandaloneFederation(
            computing,
            federation.federation_id,
            federation.parties.local.tuple(),
            [p.tuple() for p in federation.parties.parties],
        )

    else:
        raise ValueError(f"conf.federation.engine={engine} not support")
