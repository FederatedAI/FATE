def load_computing(computing):
    from fate.components.spec.computing import (
        CustomComputingSpec,
        EggrollComputingSpec,
        SparkComputingSpec,
        StandaloneComputingSpec,
    )

    if isinstance(computing, StandaloneComputingSpec):
        from fate.arch.computing.standalone import CSession

        return CSession(computing.metadata.computing_id)
    if isinstance(computing, EggrollComputingSpec):
        from fate.arch.computing.eggroll import CSession

        return CSession(computing.metadata.computing_id)
    if isinstance(computing, SparkComputingSpec):
        from fate.arch.computing.spark import CSession

        return CSession(computing.metadata.computing_id)

    # TODO: load from plugin
    raise ValueError(f"conf.computing={computing} not support")
