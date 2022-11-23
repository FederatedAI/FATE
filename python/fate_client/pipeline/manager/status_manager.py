from ml_metadata import metadata_store
from ml_metadata.proto import metadata_store_pb2


class SQLiteStatusManager(object):
    def __init__(self, status_uri: str):
        self._meta_manager = MachineLearningMetadata(metadata=dict(filename_uri=status_uri))

    @classmethod
    def create_status_manager(cls, status_uri):
        return SQLiteStatusManager(status_uri)

    def monitor_finish_status(self, execution_ids: list):
        for execution_id in execution_ids:
            task_run = self._meta_manager.get_or_create_task(execution_id)
            state = task_run.properties["state"].string_value
            if state == "running":
                return False

        return True

    def record_terminate_status(self, execution_ids):
        for execution_id in execution_ids:
            task_run = self._meta_manager.get_or_create_task(execution_id)
            self._meta_manager.set_task_safe_terminate_flag(task_run)

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

            task_run = self._meta_manager.get_or_create_task(task_info.execution_id)
            status = task_run.properties["state"].string_value

            summary_msg[role][party_id] = status
            if status != "finish":
                summary_status = "fail"

        ret = dict(summary_status=summary_status,
                   retmsg=summary_msg)

        return ret


class MachineLearningMetadata:
    def __init__(self, backend="sqlite", metadata={}) -> None:
        self.store = self.create_store(backend, metadata)
        self._job_type_id = None  # context type
        self._task_type_id = None  # execution type

    def update_task_state(self, task_run, state, exception=None):
        task_run.properties["state"].string_value = state
        if exception is not None:
            task_run.properties["exception"].string_value = exception
        self.store.put_executions([task_run])

    def get_task_safe_terminate_flag(self, task_run):
        task_run = self.get_or_create_task(task_run.name)
        return task_run.properties["safe_terminate"].bool_value

    def set_task_safe_terminate_flag(self, task_run):
        task_run.properties["safe_terminate"].bool_value = True
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

    @classmethod
    def create_store(cls, backend, metadata):
        connection_config = metadata_store_pb2.ConnectionConfig()
        if backend == "sqlite":
            connection_config.sqlite.filename_uri = metadata["filename_uri"]
            connection_config.sqlite.connection_mode = metadata.get("connection_mode", 3)
        return metadata_store.MetadataStore(connection_config)

    @property
    def job_type_id(self):
        if self._job_type_id is None:
            job_type = metadata_store_pb2.ContextType()
            job_type.name = "Job"
            job_type.properties["jobid"] = metadata_store_pb2.STRING
            self._job_type_id = self.store.put_context_type(job_type)
        return self._job_type_id

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


def get_status_manager():
    return SQLiteStatusManager
