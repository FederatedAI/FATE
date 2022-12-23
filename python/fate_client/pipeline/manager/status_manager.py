import json
from ml_metadata import metadata_store
from ml_metadata.proto import metadata_store_pb2


class SQLiteStatusManager(object):
    def __init__(self, status_uri: str):
        self._meta_manager = MachineLearningMetadata(metadata=dict(filename_uri=status_uri))

    @classmethod
    def create_status_manager(cls, status_uri):
        return SQLiteStatusManager(status_uri)

    def monitor_finish_status(self, task_ids: list):
        for task_id in task_ids:
            task_run = self._meta_manager.get_or_create_task(task_id)
            state = task_run.properties["state"].string_value
            if state in ["INIT", "running"]:
                return False

        return True

    def record_task_status(self, task_id, status):
        self._meta_manager.update_task_state(task_id, status)

    def record_terminate_status(self, task_ids):
        for task_id in task_ids:
            # task_run = self._meta_manager.get_or_create_task(execution_id)
            self._meta_manager.set_task_safe_terminate_flag(task_id)

    def get_task_results(self, tasks_info):
        """
        running/finish/exception
        """
        summary_msg = dict()
        summary_status = "success"

        for task_info in tasks_info:
            role = task_info.role
            party_id = task_info.party_id
            if role not in summary_msg:
                summary_msg[role] = dict()

            task_run = self._meta_manager.get_or_create_task(task_info.task_id)
            status = task_run.properties["state"].string_value

            summary_msg[role][party_id] = status
            if status != "finish":
                summary_status = "fail"

        ret = dict(summary_status=summary_status,
                   retmsg=summary_msg)

        return ret

    def get_task_outputs(self, task_id):
        return self._meta_manager.get_artifacts(task_id)


def get_status_manager():
    return SQLiteStatusManager


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
        data = []
        model = []
        metric = []
        for artifact in artifacts:
            if self.parameter_type_id == artifact.type_id:
                parameters.append(
                    dict(
                        name=artifact.properties["name"].string_value,
                        value=json.loads(artifact.properties["value"].string_value),
                        type=artifact.properties["type"].string_value,
                    )
                )
            if self.data_type_id == artifact.type_id:
                data.append(
                    dict(
                        uri=artifact.uri,
                        name=artifact.properties["name"].string_value,
                        metadata=json.loads(artifact.properties["metadata"].string_value),
                    )
                )

            if self.model_type_id == artifact.type_id:
                data.append(
                    dict(
                        uri=artifact.uri,
                        name=artifact.properties["name"].string_value,
                        metadata=json.loads(artifact.properties["metadata"].string_value),
                    )
                )

            if self.metric_type_id == artifact.type_id:
                data.append(
                    dict(
                        uri=artifact.uri,
                        name=artifact.properties["name"].string_value,
                        metadata=json.loads(artifact.properties["metadata"].string_value),
                    )
                )

        return dict(parameters=parameters, data=data, model=model, metric=metric)

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

    def add_data_artifact(self, name: str, uri: str, metadata: dict):
        return self.add_artifact(self.data_type_id, name, uri, metadata)

    def add_model_artifact(self, name: str, uri: str, metadata: dict):
        return self.add_artifact(self.model_type_id, name, uri, metadata)

    def add_metric_artifact(self, name: str, uri: str, metadata: dict):
        return self.add_artifact(self.metric_type_id, name, uri, metadata)

    def add_artifact(self, type_id: int, name: str, uri: str, metadata: dict):
        artifact = metadata_store_pb2.Artifact()
        artifact.uri = uri
        artifact.properties["name"].string_value = name
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
        artifact_type.properties["metadata"] = metadata_store_pb2.STRING
        artifact_type_id = self.store.put_artifact_type(artifact_type)
        return artifact_type_id
