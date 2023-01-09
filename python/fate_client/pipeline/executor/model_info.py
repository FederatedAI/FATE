from typing import Union, Dict
from ..scheduler.runtime_constructor import RuntimeConstructor


class StandaloneModelInfo(object):
    def __init__(self, job_id: str, task_info: Dict[str, RuntimeConstructor],
                 model_id: str = None, model_version: int = None):
        self._job_id = job_id
        self._task_info = task_info
        self._model_id = model_id
        self._model_version = model_version

    @property
    def job_id(self):
        return self._job_id

    @property
    def task_info(self):
        return self._task_info

    @property
    def model_id(self):
        return self._model_id

    @property
    def model_version(self):
        return self._model_version


class FateFlowModelInfo(object):
    def __init__(self, job_id: str, schedule_role: str, schedule_party_id: str,
                 model_id: str = None, model_version: int = None):
        self._job_id = job_id
        self._schedule_role = schedule_role
        self._schedule_party_id = schedule_party_id
        self._model_id = model_id
        self._model_version = model_version

    @property
    def job_id(self):
        return self._job_id

    @property
    def schedule_role(self):
        return self._schedule_role

    @property
    def schedule_party_id(self):
        return self._schedule_party_id

    @property
    def model_id(self):
        return self._model_id

    @property
    def model_version(self):
        return self._model_version
