from ..utils.session_id_gen import get_session_id

"""
execute with python -m fate.components.runner --session_id xxx --type xxx --address xxx
"""


class FateStandaloneExecutor(object):
    @classmethod
    def exec(cls, dag, job_type="train"):
        session_id = get_session_id()

        # validate param first
        cls.param_validate(dag, session_id)

        # execute nodes
        cls.run(dag, session_id)

    @classmethod
    def param_validate(cls, dag, session_id):
        for node_name in dag.topological_sort():
            node = dag.get_node(node_name)
            if getattr(node, "param_validate", True):
                node_parties = dag.get_executable_node_info(node_name)

    @classmethod
    def run(cls, dag, session_id):
        ...

    @classmethod
    def clean(cls, dag, session_id):
        ...


class FateFlowStandaloneExecutor(object):
    @classmethod
    def exec(cls, dag, job_type="train"):
        dsl = dag.get_job_dsl()
        conf = dag.get_job_conf()

        ...
