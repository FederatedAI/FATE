import logging
from typing import Protocol

from fate.components.spec.mlmd import CustomMLMDSpec, FlowMLMDSpec, PipelineMLMDSpec

logger = logging.getLogger(__name__)


def load_mlmd(mlmd, taskid):
    # from buildin
    if isinstance(mlmd, PipelineMLMDSpec):
        return PipelineMLMD(mlmd, taskid)

    if isinstance(mlmd, FlowMLMDSpec):
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


class MLMD(Protocol):
    def log_excution_start(self):
        ...

    def log_excution_end(self):
        ...

    def log_excution_exception(self, message: dict):
        ...

    def safe_terminate(self):
        ...


class PipelineMLMD:
    def __init__(self, mlmd: PipelineMLMDSpec, taskid) -> None:
        from fate.arch.context._mlmd import MachineLearningMetadata

        self._mlmd = MachineLearningMetadata(metadata=dict(filename_uri=mlmd.metadata.db))
        self._taskid = taskid

    def log_excution_start(self):
        return self._log_state("running")

    def log_excution_end(self):
        return self._log_state("finish")

    def log_excution_exception(self, message: dict):
        import json

        self._log_state("exception", json.dumps(message))

    def _log_state(self, state, message=None):
        self._mlmd.update_task_state(self._taskid, state, message)

    def safe_terminate(self):
        return self._mlmd.get_task_safe_terminate_flag(self._taskid)


class FlowMLMD:
    def __init__(self, mlmd: FlowMLMDSpec, taskid) -> None:
        self._mlmd = mlmd
        self._taskid = taskid

    def log_excution_start(self):
        return self._log_state("running")

    def log_excution_end(self):
        return self._log_state("finish")

    def log_excution_exception(self, message: dict):
        return self._log_state("exception", message)

    def _log_state(self, state, message=None):
        import requests

        data = {"state": state}
        if message is not None:
            data["message"] = message
        requests.post(self._mlmd.metadata.entrypoint, json=data)

    def safe_terminate(self):
        ...
