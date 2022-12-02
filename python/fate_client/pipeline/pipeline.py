from typing import Union
from .executor import StandaloneExecutor, FateFlowExecutor
from .entity import DAG
from .entity.runtime_entity import Roles
from .conf.types import SupportRole
from .conf.job_configuration import JobConf
from .components.component_base import Component
from .scheduler.dag_parser import DagParser
import yaml


class Pipeline(object):
    def __init__(self, executor, *args):
        self._executor = executor
        self._dag = DAG()
        self._roles = Roles()
        self._stage = "train"
        self._tasks = dict()
        self._job_conf = JobConf()
        self._model_info = None
        self._predict_dag = None

    def set_stage(self, stage):
        self._stage = stage
        return self

    def set_scheduler_party_id(self, party_id: Union[str, int]):
        self._roles.set_scheduler_party_id(party_id)
        return self

    @property
    def conf(self):
        return self._job_conf

    def set_predict_dag(self, predict_dag):
        self._predict_dag = predict_dag

    def set_roles(self, guest=None, host=None, arbiter=None, **kwargs):
        local_vars = locals()
        local_vars.pop("kwargs")
        if kwargs:
            local_vars.update(kwargs)

        support_roles = SupportRole.support_roles()
        for role, party_id in local_vars.items():
            if role == "self" or party_id is None:
                continue

            if role not in support_roles:
                raise ValueError(f"role {role} is not support")

            if isinstance(party_id, int):
                party_id = str(party_id)
            elif isinstance(party_id, list):
                party_id = [str(_id) for _id in party_id]
            self._roles.set_role(role, party_id)

        return self

    def add_task(self, task) -> "Pipeline":
        if task.name in self._tasks:
            raise ValueError(f"Task {task.name} has been added before")

        self._tasks[task.name] = task

        return self

    def compile(self) -> "Pipeline":
        self._dag.compile(task_insts=self._tasks,
                          roles=self._roles,
                          stage=self._stage,
                          job_conf=self._job_conf.conf)
        return self

    def get_dag(self):
        return yaml.dump(self._dag.dag_spec.dict(exclude_defaults=True))

    def get_component_specs(self):
        component_specs = dict()
        for task_name, task in self._tasks.items():
            component_specs[task_name] = task.component_spec

        return component_specs

    def fit(self) -> "Pipeline":
        self._model_info = self._executor.exec(self._dag.dag_spec, self.get_component_specs())

        return self

    def predict(self):
        ...

    def deploy(self, task_list=None):
        """
        this will return predict dag IR
        if component_list is None: deploy all
        """
        if task_list:
            task_name_list = []
            for task in task_list:
                if isinstance(task, Component):
                    task_name_list.append(task.name)
                else:
                    task_name_list.append(task)
        else:
            task_name_list = [task.name for (task_name, task) in self._tasks.items()]

        self._predict_dag = DagParser.deploy(task_name_list, self._dag.dag_spec, self.get_component_specs())

        return yaml.dump(self._predict_dag.dict(exclude_defaults=True))

    def __getattr__(self, attr):
        if attr in self._tasks:
            return self._tasks[attr]

        return self.__getattribute__(attr)

    def __getitem__(self, item):
        if item not in self._tasks:
            raise ValueError(f"Component {item} has not been added in pipeline")

        return self._tasks[item]


class StandalonePipeline(Pipeline):
    def __init__(self, *args):
        super(StandalonePipeline, self).__init__(StandaloneExecutor(), *args)


class FateFlowPipeline(Pipeline):
    def __init__(self, *args):
        super(FateFlowPipeline, self).__init__(FateFlowExecutor(), *args)
