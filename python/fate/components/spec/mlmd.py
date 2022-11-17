import json
from typing import Literal, Protocol

import pydantic
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


class PipelineMLMD(BaseModel):
    class PipelineMLMDMetaData(BaseModel):
        state_path: str
        terminate_state_path: str

    type: Literal["pipeline"]
    metadata: PipelineMLMDMetaData

    def init(self, execution_id: str):
        ...

    def log_excution_start(self):
        return self._log_state("running")

    def log_excution_end(self):
        return self._log_state("finish")

    def log_excution_exception(self, message: dict):
        return self._log_state("exception", message)

    def _log_state(self, state, message=None):
        data = dict(state=state)
        if message is not None:
            data["message"] = message
        with open(self.metadata.state_path, "w") as f:
            json.dump(data, f)

    def safe_terminate(self):
        ...


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
