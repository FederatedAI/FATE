from typing import Protocol


class ExecutionStatus(Protocol):
    def log_excution_start(self):
        ...

    def log_excution_end(self):
        ...

    def log_excution_exception(self, message: dict):
        ...

    def safe_terminate(self):
        ...


class IOManagerProtocol:
    def log_input_parameters(self, key, value):
        ...

    def log_input_artifact(self, key, value):
        ...

    def log_output_artifact(self, key, value):
        ...


class MLMD(Protocol):
    execution_status: ExecutionStatus
    io: IOManagerProtocol
