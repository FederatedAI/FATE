from pathlib import Path
from ..conf.env_config import LogPath
from ..utils.id_gen import get_session_id
from ..utils.job_process import process_task
from ..entity.dag import FateStandaloneDAG
from ..entity.runtime_entity import FateStandaloneRuntimeEntity

"""
execute with python -m fate.components.runner --session_id xxx --type xxx --address xxx
"""


class FateStandaloneExecutor(object):
    @classmethod
    def exec(cls, dag, job_type="train"):
        session_id = get_session_id()

        log_dir_prefix = Path(LogPath.log_directory()).joinpath(session_id)
        print(f"log prefix {log_dir_prefix}")
        # validate param first
        # cls.validate(dag, session_id, log_dir_prefix)

        # execute nodes
        runtime_entity_dict = cls.run(dag, session_id, log_dir_prefix)
        return runtime_entity_dict

    @classmethod
    def validate(cls, dag, session_id, log_dir_prefix):
        task_type = "run_component"
        for node_name in dag.topological_sort():
            log_dir = log_dir_prefix.joinpath("validate_param").joinpath(node_name)
            node_conf = dag.get_node_conf(node_name)
            module = dag.get_module(node_name)
            runtime_parties = dag.get_runtime_parties(node_name)
            runtime_entity = FateStandaloneRuntimeEntity(session_id=session_id,
                                                         node_conf=node_conf,
                                                         node_name=node_name,
                                                         module=module,
                                                         job_type=task_type,
                                                         runtime_parties=runtime_parties)

            ret_msg = cls.start_task(task_type,
                                     runtime_entity,
                                     session_id,
                                     log_dir)

            if ret_msg["retcode"] != 0:
                raise ValueError(f"Validate params are failed, error msg is {ret_msg}")

    @classmethod
    def run(cls, dag: FateStandaloneDAG, session_id, log_dir_prefix):
        runtime_entity_dict = dict()
        task_type = "run_component"
        for node_name in dag.topological_sort():
            print(f"Running component {node_name}")
            log_dir = log_dir_prefix.joinpath("tasks").joinpath(node_name)
            node_conf = dag.get_node_conf(node_name)
            module = dag.get_module(node_name)
            runtime_parties = dag.get_runtime_parties(node_name)
            upstream_node_names = dag.predecessors(node_name)
            inputs = dict()
            outputs = dag.get_node_output_interface(node_name)
            for src in upstream_node_names:
                attrs = dag.get_edge_attr(src, node_name)
                src_entity = runtime_entity_dict[src]
                """
                src: hetero_lr_0
                node_name: hetero_1
                attrs: 'hetero_lr_1', {'model': {'model': 'hetero_lr_0.model'}})
                """
                for outer_key, inner_links in attrs.items():
                    for inner_key,  inner_link in inner_links.items():
                        if outer_key in ["model", "isometric_model"]:
                            upper_in = src_entity.get_model_output_uri(key=inner_link.split(".", 1)[1])
                        else:
                            upper_in = src_entity.get_data_output_uri(key=inner_link.split(".", 1)[1])

                        if outer_key not in inputs or inner_key not in inputs[outer_key]:
                            inputs[outer_key] = {
                                inner_key: dict()
                            }
                        inputs[outer_key][inner_key][inner_link] = upper_in

            runtime_entity = FateStandaloneRuntimeEntity(session_id=session_id,
                                                         node_conf=node_conf,
                                                         node_name=node_name,
                                                         module=module,
                                                         job_type=task_type,
                                                         runtime_parties=runtime_parties,
                                                         inputs=inputs,
                                                         outputs=outputs)
            runtime_entity_dict[node_name] = runtime_entity

            if module == "data_input":
                continue

            ret_msg = cls.start_task(task_type,
                                     runtime_entity,
                                     session_id,
                                     log_dir)

            # cls.clean(session_id, runtime_entity, log_dir_prefix)

            if ret_msg["summary_status"] != "SUCCESS":
                raise ValueError(f"Execute component {node_name} failed, error msg is {ret_msg}")

        print("Job Finish Successfully!!!")
        return runtime_entity_dict

    @classmethod
    def clean(cls, session_id, runtime_entity, log_dir_prefix):
        log_dir = log_dir_prefix.joinpath("clean")
        cls.start_task("clean_task",
                       runtime_entity,
                       session_id,
                       log_dir)

    @classmethod
    def start_task(cls, task_type, runtime_entity, session_id, log_dir):
        exec_cmd_prefix = [
            "python",
            "-m",
            "fate.components.entrypoint",
            session_id
        ]
        ret_msg = process_task(task_type=task_type,
                               exec_cmd_prefix=exec_cmd_prefix,
                               runtime_entity=runtime_entity,
                               log_dir=log_dir
                               )

        return ret_msg


class FateFlowExecutor(object):
    @classmethod
    def exec(cls, dag, job_type="train"):
        dsl = dag.get_job_dsl()
        conf = dag.get_job_conf()

        ...
