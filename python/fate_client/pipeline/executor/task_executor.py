from pathlib import Path
from typing import Dict
from ..conf.env_config import LogPath
from ..utils.id_gen import gen_job_id
from ..utils.job_process import process_task
from ..entity.dag_structures import DAGSchema
from ..entity.component_structures import ComponentSpec
from ..scheduler.dag_parser import DagParser
from ..scheduler.runtime_constructor import RuntimeConstructor


class StandaloneExecutor(object):
    def __init__(self):
        self._job_id = None
        self._runtime_constructor_dict: dict = dict()
        self._dag_parser = DagParser()
        self._log_dir_prefix = None

    def exec(self, dag_schema: DAGSchema, component_specs: Dict[str, ComponentSpec]):
        self._job_id = gen_job_id()
        self._log_dir_prefix = Path(LogPath.log_directory()).joinpath(self._job_id)
        print(f"log prefix {self._log_dir_prefix}")

        self._dag_parser.parse_dag(dag_schema, component_specs)
        self._run()

    def _run(self):
        runtime_constructor_dict = dict()
        for task_name in self._dag_parser.topological_sort():
            print(f"Running component {task_name}")
            log_dir = self._log_dir_prefix.joinpath("tasks").joinpath(task_name)
            task_node = self._dag_parser.get_task_node(task_name)
            stage = task_node.stage
            runtime_parties = task_node.runtime_parties
            runtime_parameters = task_node.runtime_parameters
            component_spec = task_node.component_spec
            upstream_inputs = task_node.upstream_inputs
            output_definitions = task_node.output_definitions

            runtime_constructor = RuntimeConstructor(runtime_parties=runtime_parties,
                                                     job_id=self._job_id,
                                                     task_name=task_name,
                                                     component_ref=task_node.component_ref,
                                                     stage=stage,
                                                     runtime_parameters=runtime_parameters,
                                                     log_dir=log_dir)
            runtime_constructor.construct_input_artifacts(upstream_inputs, runtime_constructor_dict, component_spec)
            runtime_constructor.construct_output_artifacts(output_definitions)
            runtime_constructor.construct_task_schedule_spec()
            runtime_constructor_dict[task_name] = runtime_constructor

            status = self._exec_task("run_component",
                                     task_name,
                                     runtime_constructor=runtime_constructor)
            if status["summary_status"] != "success":
                raise ValueError(f"run task {task_name} is failed, status is {status}")

        self._runtime_constructor_dict = runtime_constructor_dict
        print("Job Finish Successfully!!!")

    @staticmethod
    def _exec_task(task_type, task_name, runtime_constructor):
        exec_cmd_prefix = [
            "python",
            "-m",
            "fate.components",
            "component",
            "execute",
        ]

        ret_msg = process_task(task_type=task_type,
                               task_name=task_name,
                               exec_cmd_prefix=exec_cmd_prefix,
                               runtime_constructor=runtime_constructor,
                               )

        return ret_msg


class FateFlowExecutor(object):
    ...
