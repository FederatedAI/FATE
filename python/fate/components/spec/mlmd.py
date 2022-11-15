from typing import Protocol

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


class PipelineMLMD:
    def log_excution_start(self):
        ...

    def log_excution_end(self):
        ...

    def log_excution_exception(self, msg: str):
        ...


class FlowMLMD:
    def log_excution_start(self):
        ...

    def log_excution_end(self):
        ...

    def log_excution_exception(self, msg: str):
        ...


def build_mlmd():
    ...
