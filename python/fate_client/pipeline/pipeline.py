from .executor import FateStandaloneExecutor
from .utils import FateStandaloneDAG, FateFlowDAG
from .utils.runtime_entity import Roles
from .conf.types import SupportRole


class Pipeline(object):
    def __init__(self, executor, *args):
        self._executor = executor
        self._dag = None
        self._roles = Roles()

    def set_leader(self, role, party_id):
        self._roles.set_leader(role, party_id)
        return self

    def set_roles(self, guest=None, host=None, arbiter=None, **kwargs):
        support_roles = SupportRole.support_roles()
        local_vars = locals()
        for role, party_id in local_vars.items():
            if role == "self" or party_id is None:
                continue

            if role not in support_roles:
                raise ValueError(f"role {role} is not support")

            self._roles.set_role(role, party_id)

        return self

    def add_component(self, component, data=None, train_data=None,
                      validate_data=None, model=None, isometric_model=None, cache=None, **kwargs) -> "Pipeline":
        self._dag.add_node(component, data=None, train_data=None, validate_data=None,
                           model=None, isometric_model=None, cache=None, **kwargs)
        return self

    def compile(self) -> "Pipeline":
        self._dag.compile()
        return self

    def fit(self):
        self._dag.compile()
        self._executor.exec(self._dag)

    def predict(self):
        ...

    def show(self):
        self._dag.display()


class FateStandalonePipeline(Pipeline):
    def __init__(self, *args):
        super(FateStandalonePipeline).__init__(FateStandaloneExecutor(), args)
        self._dag = FateStandaloneDAG()


class FateFlowPipeline(Pipeline):
    def __init__(self, *args):
        super(FateFlowPipeline).__init__(args)
        self._dag = FateFlowDAG()


