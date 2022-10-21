from ..utils.id_gen import get_session_id
from ..utils.job_process import process_task
from ..entity.runtime_entity import FateStandaloneRuntimeEntity

"""
execute with python -m fate.components.runner --session_id xxx --type xxx --address xxx
"""


class FateStandaloneExecutor(object):
    @classmethod
    def exec(cls, dag, job_type="train"):
        session_id = get_session_id()

        # validate param first
        cls.validate(dag, session_id)

        # execute nodes
        runtime_entity_dict = cls.run(dag, session_id)
        return runtime_entity_dict

    @classmethod
    def validate(cls, dag, session_id):
        for node_name in dag.topological_sort():
            node_conf = dag.get_node_conf(node_name)
            runtime_entity = FateStandaloneRuntimeEntity(session_id=session_id,
                                                         node_conf=node_conf,
                                                         node_name=node_name)

            ret_msg = cls.start_task("validate_params",
                                     runtime_entity,
                                     session_id)

            # if ret_msg["retcode"] != 0:
            #     raise ValueError(f"Validate params are failed, error msg is {ret_msg}")

    @classmethod
    def run(cls, dag, session_id):
        runtime_entity_dict = dict()
        for node_name in dag.topological_sort():
            node_conf = dag.get_node_conf(node_name)
            upstream_node_names = dag.predecessors(node_name)
            inputs = dict()
            for src in upstream_node_names:
                attrs = dag.get_edge_attr(src, node_name)
                src_entity = runtime_entity_dict[src]
                for k, v in attrs.items():
                    if k in ["model", "isometric_model"]:
                        in_path = src_entity.get_model_output_path()
                    else:
                        in_path = src_entity.get_data_output_path()

                    if k not in inputs:
                        inputs[k] = dict()
                    inputs[k][src] = in_path

            runtime_entity = FateStandaloneRuntimeEntity(session_id=session_id,
                                                         node_conf=node_conf,
                                                         node_name=node_name,
                                                         inputs=inputs)
            runtime_entity_dict[node_name] = runtime_entity

            ret_msg = cls.start_task("execute_component",
                                     runtime_entity,
                                     session_id)

            cls.clean(session_id, runtime_entity)

            # if ret_msg["retcode"] != 0:
            #     raise ValueError(f"Execute component {node_name} failed, error msg is {ret_msg}")

        return runtime_entity_dict

    @classmethod
    def clean(cls, session_id, runtime_entity):
        cls.start_task("clean_task",
                       runtime_entity,
                       session_id)

    @classmethod
    def start_task(cls, task_type, runtime_entity, session_id):
        exec_cmd_prefix = [
            "python",
            "-m",
            "fate.components.runner",
            "--session-id",
            session_id
        ]
        ret_msg = process_task(task_type=task_type,
                               exec_cmd_prefix=exec_cmd_prefix,
                               runtime_entity=runtime_entity
                               )

        return ret_msg


class FateFlowExecutor(object):
    @classmethod
    def exec(cls, dag, job_type="train"):
        dsl = dag.get_job_dsl()
        conf = dag.get_job_conf()

        ...
