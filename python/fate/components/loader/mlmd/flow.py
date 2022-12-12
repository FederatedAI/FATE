import logging
from typing import Optional

from pydantic import BaseModel

from fate.components.loader.mlmd import MLMD
from fate.components.loader.mlmd.protocol import IOManagerProtocol
from fate.components.spec.mlmd import FlowMLMDSpec


class ExecutionStatus:
    class StateData(BaseModel):
        execution_id: str
        status: str
        error: Optional[str]

    class Status:
        RUNNING = "running"
        SUCCESS = "success"
        FAILED = "failed"

    def __init__(self, mlmd: FlowMLMDSpec, taskid) -> None:
        self._mlmd = mlmd
        self._taskid = taskid

    def log_excution_start(self):
        return self._log_state(self.Status.RUNNING)

    def log_excution_end(self):
        return self._log_state(self.Status.SUCCESS)

    def log_excution_exception(self, message: dict):
        return self._log_state(self.Status.FAILED, message)

    def _log_state(self, state, message=None):
        error = ""
        if message:
            error = message.get("exception")
        import requests
        logging.info(self._mlmd.metadata.statu_uri)
        data = self.StateData(execution_id=self._taskid, status=state, error=error).dict()
        logging.debug(f"request flow uri: {self._mlmd.metadata.statu_uri}")
        response = requests.post(self._mlmd.metadata.statu_uri, json=data)
        logging.debug(f"response: {response.text}")

    def safe_terminate(self):
        return True


class IOManager(IOManagerProtocol):
    def __init__(self, mlmd, task_id):
        self.mlmd = mlmd
        self.task_id = task_id

    def log_output_artifact(self, key, value):
        if value is None:
            return
        from fate.components import DatasetArtifact, MetricArtifact, ModelArtifact

        if isinstance(value, DatasetArtifact):
            self.log_output_data(key, value)
        elif isinstance(value, ModelArtifact):
            self.log_output_model(key, value)
        elif isinstance(value, MetricArtifact):
            self.log_output_metric(key, value)
        else:
            raise RuntimeError(f"not supported input artifact `name={key}, value={value}`")

    def log_output_data(self, key, value):
        import requests
        logging.debug(f"request flow uri: {self.mlmd.metadata.tracking_uri}")
        response = requests.post(self.mlmd.metadata.tracking_uri,
                                 json={"output_key": value.name, "meta_data": value.metadata,
                                       "execution_id": self.task_id, "uri": value.uri, "type": "data"})
        logging.debug(f"response: {response.text}")

    def log_output_model(self, key, value):
        import requests
        logging.debug(f"request flow uri: {self.mlmd.metadata.tracking_uri}")
        response = requests.post(self.mlmd.metadata.tracking_uri,
                                 json={"output_key": value.name, "meta_data": value.metadata,
                                       "execution_id": self.task_id, "uri": value.uri, "type": "model"})
        logging.debug(response.text)
        logging.debug(value)

    def log_output_metric(self, key, value):
        logging.debug(value)


class FlowMLMD(MLMD):
    def __init__(self, mlmd: FlowMLMDSpec, taskid) -> None:
        self._taskid = taskid
        self.execution_status = ExecutionStatus(mlmd, taskid)
        self.io = IOManager(mlmd=mlmd, task_id=taskid)
