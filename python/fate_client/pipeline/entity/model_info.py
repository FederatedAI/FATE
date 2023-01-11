from typing import Dict


class StandaloneModelInfo(object):
    def __init__(self, job_id: str, task_info, local_role: str, local_party_id: str,
                 model_id: str = None, model_version: int = None):
        self._job_id = job_id
        self._task_info = task_info
        self._model_id = model_id
        self._model_version = model_version
        self._local_role = local_role
        self._local_party_id = local_party_id

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

    @property
    def local_role(self):
        return self._local_role

    @property
    def local_party_id(self):
        return self._local_party_id


class FateFlowModelInfo(object):
    def __init__(self, job_id: str, local_role: str, local_party_id: str,
                 model_id: str = None, model_version: int = None):
        self._job_id = job_id
        self._local_role = local_role
        self._local_party_id = local_party_id
        self._model_id = model_id
        self._model_version = model_version

    @property
    def job_id(self):
        return self._job_id

    @property
    def local_role(self):
        return self._local_role

    @property
    def local_party_id(self):
        return self._local_party_id

    @property
    def model_id(self):
        return self._model_id

    @property
    def model_version(self):
        return self._model_version
