from typing import Literal, Protocol

from pydantic import BaseModel

"""
ML-Metadata client for logging execution state only.
events between artifacts and executions are responsed outside.
"""


class MLMD(Protocol):
    def log_excution_start(self):
        ...

    def log_excution_end(self):
        ...

    def log_excution_exception(self, msg: str):
        ...

    def safe_terminate(self):
        ...


class PipelineMLMDDesc(BaseModel):
    class PipelineMLMDMetaData(BaseModel):
        db: str

    type: Literal["pipeline"]
    metadata: PipelineMLMDMetaData


class PipelineMLMD:
    def __init__(self, mlmd: PipelineMLMDDesc, taskid) -> None:
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


class FlowMLMD(BaseModel):
    class FlowMLMDMetaData(BaseModel):
        entrypoint: str

    type: Literal["flow"]
    metadata: FlowMLMDMetaData

    def init(self, execution_id: str):
        ...

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
        requests.post(self.metadata.entrypoint, json=data)

    def safe_terminate(self):
        ...


def get_mlmd(mlmd, taskid):
    if isinstance(mlmd, PipelineMLMDDesc):
        return PipelineMLMD(mlmd, taskid)
