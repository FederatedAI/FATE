import time
from enum import Enum

from fate.components.runner import Runner
from fate.components.runner.context import ComponentContext


class TaskExecuteStatus(Enum):
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"


class TaskExecutorClient:
    def __init__(self, address) -> None:
        pass

    def fetch_task_config(self):
        ...

    def update_job_status(self, status: TaskExecuteStatus):
        ...

    def safe_to_terminate(self):
        ...

    tracker: ...


def task_execute(address):

    task_client = TaskExecutorClient(address)
    task_client.fetch_task_config()
    context = ComponentContext(...)
    try:
        task_client.update_job_status(TaskExecuteStatus.RUNNING)
        try:
            Runner(...).run(...)
        except Exception:
            task_client.update_job_status(TaskExecuteStatus.FAILED)
        else:
            task_client.update_job_status(TaskExecuteStatus.SUCCESS)

        while not task_client.safe_to_terminate():
            time.sleep(0.5)
    finally:
        context.clean()
