from .executor import FateStandaloneExecutor, FateFlowExecutor
from .entity import FateStandaloneDAG, FateFlowDAG
from .entity.runtime_entity import Roles
from .conf.types import SupportRole


class Pipeline(object):
    def __init__(self, executor):
        self._executor = executor
        self._dag = None
        self._roles = Roles()

    def set_leader(self, role, party_id):
        self._roles.set_leader(role, str(party_id))
        return self

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

    def add_component(self, component, data=None, train_data=None,
                      validate_data=None, model=None, isometric_model=None, cache=None, **kwargs) -> "Pipeline":
        self._dag.add_node(component, data=data, train_data=train_data, validate_data=validate_data,
                           model=model, isometric_model=isometric_model, cache=cache, **kwargs)
        return self

    def compile(self) -> "Pipeline":
        self._dag.set_roles(self._roles)
        self._dag.compile()
        return self

    def fit(self) -> "Pipeline":
        self.compile()
        self._executor.exec(self._dag)

        return self

    def predict(self):
        ...

    def deploy(self):
        ...

    def show(self):
        self._dag.display()


class FateStandalonePipeline(Pipeline):
    def __init__(self, *args):
        super(FateStandalonePipeline, self).__init__(FateStandaloneExecutor, *args)
        self._dag = FateStandaloneDAG()


class FateFlowPipeline(Pipeline):
    def __init__(self, *args):
        super(FateFlowPipeline, self).__init__(FateFlowExecutor, *args)
        self._dag = FateFlowDAG()
