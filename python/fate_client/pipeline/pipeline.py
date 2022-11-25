from typing import Union
from .executor import StandaloneExecutor, FateFlowExecutor
from .entity import DAG
from .entity.runtime_entity import Roles
from .conf.types import SupportRole
from .conf.job_configuration import JobConf
import yaml


class Pipeline(object):
    def __init__(self, executor, *args):
        self._executor = executor
        self._dag = DAG()
        self._roles = Roles()
        self._stage = "train"
        self._components = dict()
        self._job_conf = JobConf()

    def set_stage(self, stage):
        self._stage = stage
        return self

    def set_scheduler_party_id(self, party_id: Union[str, int]):
        self._roles.set_scheduler_party_id(party_id)
        return self

    @property
    def conf(self):
        return self._job_conf

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

    def add_component(self, component) -> "Pipeline":
        if component.name in self._components:
            raise ValueError(f"Component {component.name} has been added before")

        self._components[component.name] = component

        return self

    def compile(self) -> "Pipeline":
        self._dag.compile(components=self._components,
                          roles=self._roles,
                          stage=self._stage,
                          job_conf=self._job_conf.conf)
        return self

    def get_dag(self):
        return yaml.dump(self._dag.dag_spec.dict(exclude_defaults=True))

    def get_component_specs(self):
        component_specs = dict()
        for component_name, component in self._components.items():
            component_specs[component_name] = component.component_spec

        return component_specs

    def fit(self) -> "Pipeline":
        self._executor.exec(self._dag.dag_spec, self.get_component_specs())

        return self

    def predict(self):
        ...

    def deploy(self):
        ...

    def show(self):
        ...


class StandalonePipeline(Pipeline):
    def __init__(self, *args):
        super(StandalonePipeline, self).__init__(StandaloneExecutor(), *args)


class FateFlowPipeline(Pipeline):
    def __init__(self, *args):
        super(FateFlowPipeline, self).__init__(FateFlowExecutor(), *args)
