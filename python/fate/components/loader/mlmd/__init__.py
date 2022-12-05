import logging

from fate.components.spec.mlmd import CustomMLMDSpec, FlowMLMDSpec, PipelineMLMDSpec

from .protocol import MLMD

logger = logging.getLogger(__name__)


def load_mlmd(mlmd, taskid) -> MLMD:
    # from buildin
    if isinstance(mlmd, PipelineMLMDSpec):
        from .pipeline import PipelineMLMD

        return PipelineMLMD(mlmd, taskid)

    if isinstance(mlmd, FlowMLMDSpec):
        from .flow import FlowMLMD

        return FlowMLMD(mlmd, taskid)

    # from entrypoint
    if isinstance(mlmd, CustomMLMDSpec):
        import pkg_resources

        for mlmd_ep in pkg_resources.iter_entry_points(group="fate.ext.mlmd"):
            try:
                mlmd_register = mlmd_ep.load()
                mlmd_registered_name = mlmd_register.registered_name()
            except Exception as e:
                logger.warning(
                    f"register cpn from entrypoint(named={mlmd_ep.name}, module={mlmd_ep.module_name}) failed: {e}"
                )
                continue
            if mlmd_registered_name == mlmd.name:
                return mlmd_register
        raise RuntimeError(f"could not find registerd mlmd named `{mlmd.name}`")

    raise ValueError(f"unknown mlmd spec: `{mlmd}`")
