#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from pathlib import Path
from typing import Dict
from ..conf.env_config import StandaloneConfig
from ..utils.standalone.id_gen import gen_job_id
from ..utils.standalone.job_process import process_task
from ..entity.dag_structures import DAGSchema
from ..entity.component_structures import ComponentSpec
from ..scheduler.dag_parser import DagParser
from ..scheduler.runtime_constructor import RuntimeConstructor
from ..utils.fateflow.fate_flow_job_invoker import FATEFlowJobInvoker
from ..entity.model_info import StandaloneModelInfo, FateFlowModelInfo


class StandaloneExecutor(object):
    def __init__(self):
        self._job_id = None
        self._runtime_constructor_dict: dict = dict()
        self._dag_parser = DagParser()
        self._log_dir_prefix = None

    def fit(self, dag_schema: DAGSchema, component_specs: Dict[str, ComponentSpec],
            local_role: str, local_party_id: str) -> StandaloneModelInfo:
        self._dag_parser.parse_dag(dag_schema, component_specs)
        self._run()

        local_party_id = self.get_site_party_id(dag_schema, local_role, local_party_id)
        return StandaloneModelInfo(
            job_id=self._job_id,
            task_info=self._runtime_constructor_dict,
            local_role=local_role,
            local_party_id=local_party_id,
            model_id=self._job_id,
            model_version=0
        )

    def predict(self,
                dag_schema: DAGSchema,
                component_specs: Dict[str, ComponentSpec],
                fit_model_info: StandaloneModelInfo) -> StandaloneModelInfo:
        self._dag_parser.parse_dag(dag_schema, component_specs)
        self._run(fit_model_info)
        return StandaloneModelInfo(
            job_id=self._job_id,
            task_info=self._runtime_constructor_dict,
            local_role=fit_model_info.local_role,
            local_party_id=fit_model_info.local_party_id
        )

    def _run(self, fit_model_info: StandaloneModelInfo = None):
        self._job_id = gen_job_id()
        self._log_dir_prefix = StandaloneConfig.OUTPUT_LOG_DIR.joinpath(self._job_id)
        print(f"log prefix {self._log_dir_prefix}")

        runtime_constructor_dict = dict()
        for task_name in self._dag_parser.topological_sort():
            print(f"Running component {task_name}")
            log_dir = self._log_dir_prefix.joinpath("tasks").joinpath(task_name)
            task_node = self._dag_parser.get_task_node(task_name)
            stage = task_node.stage
            runtime_parties = task_node.runtime_parties
            runtime_parameters = task_node.runtime_parameters
            upstream_inputs = task_node.upstream_inputs

            runtime_constructor = RuntimeConstructor(runtime_parties=runtime_parties,
                                                     job_id=self._job_id,
                                                     task_name=task_name,
                                                     component_ref=task_node.component_ref,
                                                     component_spec=task_node.component_spec,
                                                     stage=stage,
                                                     runtime_parameters=runtime_parameters,
                                                     log_dir=log_dir)
            runtime_constructor.construct_input_artifacts(upstream_inputs,
                                                          runtime_constructor_dict,
                                                          fit_model_info)
            runtime_constructor.construct_outputs()
            # runtime_constructor.construct_output_artifacts(output_definitions)
            runtime_constructor.construct_task_schedule_spec()
            runtime_constructor_dict[task_name] = runtime_constructor

            status = self._exec_task("run_component",
                                     task_name,
                                     runtime_constructor=runtime_constructor)
            if status["summary_status"] != "success":
                raise ValueError(f"run task {task_name} is failed, status is {status}")

            runtime_constructor_dict[task_name].retrieval_task_outputs()

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

    @staticmethod
    def get_site_party_id(dag_schema, role, party_id):
        if party_id:
            return party_id

        if party_id is None:
            for party in dag_schema.dag.parties:
                if role == party.role:
                    return party.party_id[0]

        raise ValueError(f"Can not retrieval site's party_id from site's role {role}")


class FateFlowExecutor(object):
    def __init__(self):
        ...

    def fit(self, dag_schema: DAGSchema, component_specs: Dict[str, ComponentSpec],
            local_role: str, local_party_id: str) -> FateFlowModelInfo:
        flow_job_invoker = FATEFlowJobInvoker()
        local_party_id = self.get_site_party_id(flow_job_invoker, dag_schema, local_role, local_party_id)

        return self._run(dag_schema, local_role, local_party_id, flow_job_invoker)

    def predict(self,
                dag_schema: DAGSchema,
                component_specs: Dict[str, ComponentSpec],
                fit_model_info: FateFlowModelInfo) -> FateFlowModelInfo:
        flow_job_invoker = FATEFlowJobInvoker()
        schedule_role = fit_model_info.local_role
        schedule_party_id = fit_model_info.local_party_id

        return self._run(dag_schema, schedule_role, schedule_party_id, flow_job_invoker)

    def _run(self,
             dag_schema: DAGSchema,
             local_role,
             local_party_id,
             flow_job_invoker: FATEFlowJobInvoker) -> FateFlowModelInfo:

        job_id, model_id, model_version = flow_job_invoker.submit_job(dag_schema.dict(exclude_defaults=True))

        flow_job_invoker.monitor_status(job_id, local_role, local_party_id)

        return FateFlowModelInfo(
            job_id=job_id,
            local_role=local_role,
            local_party_id=local_party_id,
            model_id=model_id,
            model_version=model_version
        )

    @staticmethod
    def get_site_party_id(flow_job_invoker, dag_schema, role, party_id):
        """
        query it by flow, if backend is standalone, multiple party_ids exist, so need to decide it by query dag
        """
        site_party_id = flow_job_invoker.query_site_info()

        if site_party_id:
            return site_party_id

        if party_id:
            return party_id

        if site_party_id is None:
            for party in dag_schema.dag.parties:
                if role == party.role:
                    return party.party_id[0]

        raise ValueError(f"Can not retrieval site's party_id from site's role {role}")

    @staticmethod
    def upload(file: str, head: int,
               namespace: str, name: str, meta: dict,
               partitions=4, destroy=True, storage_engine=None, **kwargs):
        flow_job_invoker = FATEFlowJobInvoker()
        post_data = dict(file=file,
                         head=head,
                         namespace=namespace,
                         name=name,
                         meta=meta,
                         partitions=partitions,
                         destroy=destroy)
        if storage_engine:
            post_data["storage_engine"] = storage_engine

        if kwargs:
            post_data.update(kwargs)

        flow_job_invoker.upload_data(post_data)
