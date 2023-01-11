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
import json

from ml_metadata import metadata_store
from ml_metadata.proto import metadata_store_pb2


class MachineLearningMetadata:
    def __init__(self, backend="sqlite", metadata={}) -> None:
        self.store = self.create_store(backend, metadata)
        self._task_context_type_id = None  # context type
        self._task_type_id = None  # execution type
        self._data_type_id = None  # data artifact
        self._model_type_id = None  # model artifact
        self._metric_type_id = None  # metric artifact
        self._parameter_type_id = None  # parameter artifact

    @classmethod
    def create_store(cls, backend, metadata):
        connection_config = metadata_store_pb2.ConnectionConfig()
        if backend == "sqlite":
            connection_config.sqlite.filename_uri = metadata["filename_uri"]
            connection_config.sqlite.connection_mode = metadata.get("connection_mode", 3)
        return metadata_store.MetadataStore(connection_config)

    def get_artifacts(self, taskid):
        context_id = self.get_or_create_task_context(taskid).id
        artifacts = self.store.get_artifacts_by_context(context_id)
        # parameters
        parameters = []
        input_data, output_data = [], []
        input_model, output_model = [], []
        input_metric, output_metric = [], []

        def _to_dict(artifact):
            return dict(
                uri=artifact.uri,
                name=artifact.properties["name"].string_value,
                metadata=json.loads(artifact.properties["metadata"].string_value),
            )

        for artifact in artifacts:
            if self.parameter_type_id == artifact.type_id:
                parameters.append(
                    dict(
                        name=artifact.properties["name"].string_value,
                        value=json.loads(artifact.properties["value"].string_value),
                        type=artifact.properties["type"].string_value,
                    )
                )
            if artifact.type_id in {self.data_type_id, self.model_type_id, self.metric_type_id}:
                is_input = artifact.properties["is_input"].bool_value

                if self.data_type_id == artifact.type_id:
                    if is_input:
                        input_data.append(_to_dict(artifact))
                    else:
                        output_data.append(_to_dict(artifact))

                if self.model_type_id == artifact.type_id:
                    if is_input:
                        input_model.append(_to_dict(artifact))
                    else:
                        output_model.append(_to_dict(artifact))

                if self.metric_type_id == artifact.type_id:
                    if is_input:
                        input_metric.append(_to_dict(artifact))
                    else:
                        output_metric.append(_to_dict(artifact))
        return dict(
            parameters=parameters,
            input=dict(data=input_data, model=input_model, metric=input_metric),
            output=dict(data=output_data, model=output_model, metric=output_metric),
        )

    def get_or_create_task_context(self, taskid):
        task_context_run = self.store.get_context_by_type_and_name("TaskContext", taskid)
        if task_context_run is None:
            task_context_run = metadata_store_pb2.Context()
            task_context_run.type_id = self.task_context_type_id
            task_context_run.name = taskid
        [task_context_run_id] = self.store.put_contexts([task_context_run])
        task_context_run.id = task_context_run_id
        return task_context_run

    def put_task_to_task_context(self, taskid):
        association = metadata_store_pb2.Association()
        association.execution_id = self.get_or_create_task(taskid).id
        association.context_id = self.get_or_create_task_context(taskid).id
        self.store.put_attributions_and_associations([], [association])

    def put_artifact_to_task_context(self, taskid, artifact_id):
        attribution = metadata_store_pb2.Attribution()
        attribution.artifact_id = artifact_id
        attribution.context_id = self.get_or_create_task_context(taskid).id
        self.store.put_attributions_and_associations([attribution], [])

    def update_task_state(self, taskid, state, exception=None):
        task_run = self.get_or_create_task(taskid)
        task_run.properties["state"].string_value = state
        if exception is not None:
            task_run.properties["exception"].string_value = exception
        self.store.put_executions([task_run])

    def get_or_create_task(self, taskid):
        task_run = self.store.get_execution_by_type_and_name("Task", taskid)
        if task_run is None:
            task_run = metadata_store_pb2.Execution()
            task_run.type_id = self.task_type_id
            task_run.name = taskid
            task_run.properties["state"].string_value = "INIT"
            task_run.properties["safe_terminate"].bool_value = False
            [task_run_id] = self.store.put_executions([task_run])
            task_run.id = task_run_id
        return task_run

    def get_task_safe_terminate_flag(self, taskid: str):
        task_run = self.get_or_create_task(taskid)
        return task_run.properties["safe_terminate"].bool_value

    def set_task_safe_terminate_flag(self, taskid: str):
        task_run = self.get_or_create_task(taskid)
        task_run.properties["safe_terminate"].bool_value = True
        self.store.put_executions([task_run])

    def record_input_event(self, execution_id, artifact_id):
        event = metadata_store_pb2.Event()
        event.artifact_id = artifact_id
        event.execution_id = execution_id
        event.type = metadata_store_pb2.Event.DECLARED_INPUT
        self.store.put_events([event])

    def record_output_event(self, execution_id, artifact_id):
        event = metadata_store_pb2.Event()
        event.artifact_id = artifact_id
        event.execution_id = execution_id
        event.type = metadata_store_pb2.Event.DECLARED_OUTPUT
        self.store.put_events([event])

    def add_parameter(self, name: str, value):
        artifact = metadata_store_pb2.Artifact()
        artifact.properties["name"].string_value = name
        artifact.properties["type"].string_value = str(type(value))
        artifact.properties["value"].string_value = json.dumps(value)
        artifact.type_id = self.parameter_type_id
        [artifact_id] = self.store.put_artifacts([artifact])
        return artifact_id

    def add_data_artifact(self, name: str, uri: str, metadata: dict, is_input):
        return self.add_artifact(self.data_type_id, name, uri, metadata, is_input)

    def add_model_artifact(self, name: str, uri: str, metadata: dict, is_input):
        return self.add_artifact(self.model_type_id, name, uri, metadata, is_input)

    def add_metric_artifact(self, name: str, uri: str, metadata: dict, is_input):
        return self.add_artifact(self.metric_type_id, name, uri, metadata, is_input)

    def add_artifact(self, type_id: int, name: str, uri: str, metadata: dict, is_input):
        artifact = metadata_store_pb2.Artifact()
        artifact.uri = uri
        artifact.properties["name"].string_value = name
        artifact.properties["is_input"].bool_value = is_input
        artifact.properties["metadata"].string_value = json.dumps(metadata)
        artifact.type_id = type_id
        [artifact_id] = self.store.put_artifacts([artifact])
        return artifact_id

    @property
    def task_context_type_id(self):
        if self._task_context_type_id is None:
            job_type = metadata_store_pb2.ContextType()
            job_type.name = "TaskContext"
            job_type.properties["jobid"] = metadata_store_pb2.STRING
            self._task_context_type_id = self.store.put_context_type(job_type)
        return self._task_context_type_id

    @property
    def task_type_id(self):
        if self._task_type_id is None:
            task_type = metadata_store_pb2.ExecutionType()
            task_type.name = "Task"
            task_type.properties["state"] = metadata_store_pb2.STRING
            task_type.properties["exception"] = metadata_store_pb2.STRING
            task_type.properties["safe_terminate"] = metadata_store_pb2.BOOLEAN
            self._task_type_id = self.store.put_execution_type(task_type)
        return self._task_type_id

    @property
    def parameter_type_id(self):
        if self._parameter_type_id is None:
            artifact_type = metadata_store_pb2.ArtifactType()
            artifact_type.name = "Parameter"
            artifact_type.properties["name"] = metadata_store_pb2.STRING
            artifact_type.properties["type"] = metadata_store_pb2.STRING
            artifact_type.properties["value"] = metadata_store_pb2.STRING
            self._parameter_type_id = self.store.put_artifact_type(artifact_type)
        return self._parameter_type_id

    @property
    def data_type_id(self):
        if self._data_type_id is None:
            self._data_type_id = self.create_artifact_type("Data")
        return self._data_type_id

    @property
    def model_type_id(self):
        if self._model_type_id is None:
            self._model_type_id = self.create_artifact_type("Model")
        return self._model_type_id

    @property
    def metric_type_id(self):
        if self._metric_type_id is None:
            self._metric_type_id = self.create_artifact_type("Metric")
        return self._metric_type_id

    def create_artifact_type(self, name):
        artifact_type = metadata_store_pb2.ArtifactType()
        artifact_type.name = name
        artifact_type.properties["uri"] = metadata_store_pb2.STRING
        artifact_type.properties["name"] = metadata_store_pb2.STRING
        artifact_type.properties["is_input"] = metadata_store_pb2.BOOLEAN
        artifact_type.properties["metadata"] = metadata_store_pb2.STRING
        artifact_type_id = self.store.put_artifact_type(artifact_type)
        return artifact_type_id
