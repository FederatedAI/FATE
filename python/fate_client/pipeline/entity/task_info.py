import abc
import typing
from .model_info import StandaloneModelInfo, FateFlowModelInfo
from ..utils.fateflow.fate_flow_job_invoker import FATEFlowJobInvoker


class TaskInfo(object):
    def __init__(self, task_name: str, model_info: typing.Union[StandaloneModelInfo, FateFlowModelInfo]):
        self._model_info = model_info
        self._task_name = task_name

    @abc.abstractmethod
    def get_output_data(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def get_output_model(self, *args, **kwargs):
        ...

    @abc.abstractmethod
    def get_output_metrics(self, *args, **kwargs):
        ...


class StandaloneTaskInfo(TaskInfo):
    def get_output_data(self, role=None, party_id=None):
        party_id = party_id if role else self._model_info.local_party_id
        role = role if role else self._model_info.local_role
        return self._model_info.task_info[self._task_name].get_output_data(role, party_id)

    def get_output_model(self, role=None, party_id=None):
        party_id = party_id if role else self._model_info.local_party_id
        role = role if role else self._model_info.local_role
        return self._model_info.task_info[self._task_name].get_output_model(role, party_id)

    def get_output_metrics(self, role=None, party_id=None):
        party_id = party_id if role else self._model_info.local_party_id
        role = role if role else self._model_info.local_role
        return self._model_info.task_info[self._task_name].get_output_metrics(role, party_id)


class FateFlowTaskInfo(TaskInfo):
    def get_output_model(self):
        return FATEFlowJobInvoker().get_output_model(job_id=self._model_info.job_id,
                                                     role=self._model_info.local_role,
                                                     party_id=self._model_info.local_party_id,
                                                     task_name=self._task_name)

    def get_output_data(self, limits=None, ):
        raise ValueError("fate-flow does not support get_output_data interface this version.")

    def get_output_metrics(self):
        return FATEFlowJobInvoker().get_output_metrics(job_id=self._model_info.job_id,
                                                       role=self._model_info.local_role,
                                                       party_id=self._model_info.local_party_id,
                                                       task_name=self._task_name)



