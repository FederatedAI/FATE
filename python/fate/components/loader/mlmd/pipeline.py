from fate.arch.context._mlmd import MachineLearningMetadata
from fate.components.spec.mlmd import PipelineMLMDSpec

from .protocol import MLMD
from .protocol import ExecutionStatus as ExecutionStatusProtocol
from .protocol import IOManagerProtocol


class PipelineMLMD(MLMD):
    def __init__(self, mlmd: PipelineMLMDSpec, taskid) -> None:
        self._mlmd = MachineLearningMetadata(metadata=dict(filename_uri=mlmd.metadata.db))
        self.execution_status = ExecutionStatus(self._mlmd, taskid)
        self._taskid = taskid
        self.io = IOManager()


class IOManager(IOManagerProtocol):
    def log_input_parameters(self, key, value):
        ...

    def log_input_artifact(self, key, value):
        from fate.components import DatasetArtifact, MetricArtifact, ModelArtifact

        if isinstance(value, DatasetArtifact):
            self.log_input_data(key, value)
        elif isinstance(value, ModelArtifact):
            self.log_input_model(key, value)
        elif isinstance(value, MetricArtifact):
            self.log_input_metric(key, value)
        else:
            raise RuntimeError(f"not supported input artifact `name={key}, value={value}`")

    def log_output_artifact(self, key, value):
        from fate.components import DatasetArtifact, MetricArtifact, ModelArtifact

        if isinstance(value, DatasetArtifact):
            self.log_output_data(key, value)
        elif isinstance(value, ModelArtifact):
            self.log_output_model(key, value)
        elif isinstance(value, MetricArtifact):
            self.log_output_metric(key, value)
        else:
            raise RuntimeError(f"not supported input artifact `name={key}, value={value}`")

    def log_input_data(self, key, value):
        ...

    def log_input_model(self, key, value):
        ...

    def log_input_metric(self, key, value):
        ...

    def log_output_data(self, key, value):
        ...

    def log_output_model(self, key, value):
        ...

    def log_output_metric(self, key, value):
        ...


class ExecutionStatus(ExecutionStatusProtocol):
    def __init__(self, mlmd: MachineLearningMetadata, taskid) -> None:
        self._mlmd = mlmd
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
