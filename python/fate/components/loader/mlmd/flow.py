from fate.components.spec.mlmd import FlowMLMDSpec


class ExecutionStatus:
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


class IOManagerProtocol:
    ...


class FlowMLMD:
    execution_status: ExecutionStatus
    io: IOManagerProtocol
