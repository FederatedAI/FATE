#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from fate.arch.context._mlmd import MachineLearningMetadata
from fate.components.spec.mlmd import PipelineMLMDSpec

from .protocol import MLMD
from .protocol import ExecutionStatus as ExecutionStatusProtocol
from .protocol import IOManagerProtocol


class PipelineMLMD(MLMD):
    def __init__(self, mlmd: PipelineMLMDSpec, taskid) -> None:
        self._mlmd = MachineLearningMetadata(metadata=dict(filename_uri=mlmd.metadata.db))
        self._taskid = taskid
        self.execution_status = ExecutionStatus(self._mlmd, self._taskid)
        self.io = IOManager(self._mlmd, self._taskid)


class IOManager(IOManagerProtocol):
    def __init__(self, mlmd: MachineLearningMetadata, taskid) -> None:
        self._mlmd = mlmd
        self._taskid = taskid

    def log_input_artifact(self, key, value):
        if value is None:
            return
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

    def log_input_parameter(self, key, value):
        artifact_id = self._mlmd.add_parameter(name=key, value=value)
        execution_id = self._mlmd.get_or_create_task(self._taskid).id
        self._mlmd.record_input_event(execution_id=execution_id, artifact_id=artifact_id)
        self._mlmd.put_artifact_to_task_context(self._taskid, artifact_id)

    def log_input_data(self, key, value):
        artifact_id = self._mlmd.add_data_artifact(
            name=value.name, uri=value.uri, metadata=value.metadata, is_input=True
        )
        execution_id = self._mlmd.get_or_create_task(self._taskid).id
        self._mlmd.record_input_event(execution_id=execution_id, artifact_id=artifact_id)
        self._mlmd.put_artifact_to_task_context(self._taskid, artifact_id)

    def log_input_model(self, key, value):
        artifact_id = self._mlmd.add_model_artifact(
            name=value.name, uri=value.uri, metadata=value.metadata, is_input=True
        )
        execution_id = self._mlmd.get_or_create_task(self._taskid).id
        self._mlmd.record_input_event(execution_id=execution_id, artifact_id=artifact_id)
        self._mlmd.put_artifact_to_task_context(self._taskid, artifact_id)

    def log_input_metric(self, key, value):
        artifact_id = self._mlmd.add_metric_artifact(
            name=value.name, uri=value.uri, metadata=value.metadata, is_input=True
        )
        execution_id = self._mlmd.get_or_create_task(self._taskid).id
        self._mlmd.record_input_event(execution_id=execution_id, artifact_id=artifact_id)
        self._mlmd.put_artifact_to_task_context(self._taskid, artifact_id)

    def log_output_data(self, key, value):
        artifact_id = self._mlmd.add_data_artifact(
            name=value.name, uri=value.uri, metadata=value.metadata, is_input=False
        )
        execution_id = self._mlmd.get_or_create_task(self._taskid).id
        self._mlmd.record_output_event(execution_id=execution_id, artifact_id=artifact_id)
        self._mlmd.put_artifact_to_task_context(self._taskid, artifact_id)

    def log_output_model(self, key, value, metadata={}):
        artifact_id = self._mlmd.add_model_artifact(
            name=value.name, uri=value.uri, metadata=value.metadata, is_input=False
        )
        execution_id = self._mlmd.get_or_create_task(self._taskid).id
        self._mlmd.record_output_event(execution_id=execution_id, artifact_id=artifact_id)
        self._mlmd.put_artifact_to_task_context(self._taskid, artifact_id)

    def log_output_metric(self, key, value):
        artifact_id = self._mlmd.add_metric_artifact(
            name=value.name, uri=value.uri, metadata=value.metadata, is_input=False
        )
        execution_id = self._mlmd.get_or_create_task(self._taskid).id
        self._mlmd.record_output_event(execution_id=execution_id, artifact_id=artifact_id)
        self._mlmd.put_artifact_to_task_context(self._taskid, artifact_id)


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
