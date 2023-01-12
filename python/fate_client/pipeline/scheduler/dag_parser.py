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
import copy
import networkx as nx

from pydantic import BaseModel
from typing import Dict, List, Union

from ..conf.types import ArtifactType, JobStage, InputDataKeyType, ArtifactSourceType
from ..entity.dag_structures import DAGSchema, DAGSpec, RuntimeTaskOutputChannelSpec, ModelWarehouseChannelSpec
from ..entity.component_structures import ArtifactSpec, ComponentSpec


class DagParser(object):
    def __init__(self):
        self._dag = nx.DiGraph()
        self._links = dict()
        self._task_parameters = dict()
        self._task_parties = dict()
        self._tasks = dict()
        self._conf = dict()

    def parse_dag(self, dag_schema: DAGSchema, component_specs: Dict[str, ComponentSpec] = None):
        dag_spec = dag_schema.dag
        dag_stage = dag_spec.stage
        tasks = dag_spec.tasks
        if dag_spec.conf:
            self._conf = dag_spec.conf.dict(exclude_defaults=True)
        job_conf = self._conf.get("task", {})
        for name, task_spec in tasks.items():
            self._dag.add_node(name)
            task_stage = dag_stage
            component_ref = task_spec.component_ref
            if not task_spec.conf:
                task_conf = copy.deepcopy(job_conf)
            else:
                task_conf = copy.deepcopy(task_spec.conf).update(job_conf)
            if task_spec.stage:
                task_stage = task_spec.stage

            self._tasks[name] = TaskNodeInfo()
            self._tasks[name].stage = task_stage
            self._tasks[name].component_ref = component_ref
            if component_specs:
                self._tasks[name].component_spec = component_specs[name]
            self._init_task_runtime_parameters_and_conf(name, dag_schema, task_conf)

            if not task_spec.inputs or not task_spec.inputs.artifacts:
                continue

            upstream_inputs = dict()
            runtime_roles = self._tasks[name].runtime_roles
            for input_key, output_specs_dict in task_spec.inputs.artifacts.items():
                upstream_inputs[input_key] = dict()
                for artifact_source, channel_spec_list in output_specs_dict.items():
                    if artifact_source == ArtifactSourceType.MODEL_WAREHOUSE:
                        if isinstance(channel_spec_list, list):
                            inputs = []
                            for channel in channel_spec_list:
                                model_warehouse_channel = ModelWarehouseChannelSpec(**channel.dict(exclude_defaults=True))
                                if model_warehouse_channel.model_id is None:
                                    model_warehouse_channel.model_id = self._conf.get("model_id", None)
                                    model_warehouse_channel.model_version = self._conf.get("model_version", None)
                                inputs.append(model_warehouse_channel)
                        else:
                            inputs = ModelWarehouseChannelSpec(**channel_spec_list.dict(exclude_defaults=True))
                            if inputs.model_id is None:
                                inputs.model_id = self._conf.get("model_id", None)
                                inputs.model_version = self._conf.get("model_version", None)

                        upstream_inputs[input_key] = inputs
                        continue
                    else:
                        if isinstance(channel_spec_list, list):
                            inputs = [RuntimeTaskOutputChannelSpec(**channel.dict(exclude_defaults=True))
                                      for channel in channel_spec_list]
                        else:
                            inputs = RuntimeTaskOutputChannelSpec(**channel_spec_list.dict(exclude_defaults=True))

                        upstream_inputs[input_key] = inputs

                    if not isinstance(channel_spec_list, list):
                        channel_spec_list = [channel_spec_list]

                    for channel_spec in channel_spec_list:
                        dependent_task = channel_spec.producer_task
                        self._add_edge(dependent_task, name)

            upstream_inputs = self.check_and_add_runtime_roles(upstream_inputs, runtime_roles)
            self._tasks[name].upstream_inputs = upstream_inputs

    def _add_edge(self, src, dst, attrs=None):
        if not attrs:
            attrs = {}

        self._dag.add_edge(src, dst, **attrs)

    def _init_task_runtime_parameters_and_conf(self, task_name: str, dag_schema: DAGSchema, global_task_conf):
        dag = dag_schema.dag
        role_keys = set([party.role for party in dag.parties])
        task_spec = dag.tasks[task_name]
        if task_spec.parties:
            task_role_keys = set([party.role for party in task_spec.parties])
            role_keys = role_keys & task_role_keys

        common_parameters = dict()
        if task_spec.inputs and task_spec.inputs.parameters:
            common_parameters = task_spec.inputs.parameters

        task_parameters = dict()
        task_conf = dict()
        task_runtime_parties = []

        for party in dag.parties:
            if party.role not in role_keys:
                continue
            task_parameters[party.role] = dict()
            task_conf[party.role] = dict()
            for party_id in party.party_id:
                task_parameters[party.role][party_id] = copy.deepcopy(common_parameters)
                task_conf[party.role][party_id] = copy.deepcopy(global_task_conf)
                task_runtime_parties.append(Party(role=party.role, party_id=party_id))

        if dag.party_tasks:
            party_tasks = dag.party_tasks
            for site_name, party_tasks_spec in party_tasks.items():
                if task_name not in party_tasks_spec.tasks:
                    continue

                party_task_conf = copy.deepcopy(party_tasks_spec.conf) if party_tasks_spec.conf else dict()
                party_task_conf.update(global_task_conf)

                party_parties = party_tasks_spec.parties
                party_task_spec = party_tasks_spec.tasks[task_name]

                if party_task_spec.conf:
                    _conf = copy.deepcopy(party_task_spec.conf)
                    party_task_conf = _conf.update(party_task_conf)
                for party in party_parties:
                    if party.role in task_parameters:
                        for party_id in party.party_id:
                            task_conf[party.role][party_id].update(party_task_conf)

                if not party_task_spec.inputs:
                    continue
                parameters = party_task_spec.inputs.parameters

                if parameters:
                    for party in party_parties:
                        if party.role in task_parameters:
                            for party_id in party.party_id:
                                task_parameters[party.role][party_id].update(parameters)

        self._tasks[task_name].runtime_parameters = task_parameters
        self._tasks[task_name].runtime_parties = task_runtime_parties
        self._tasks[task_name].conf = task_conf

    def get_runtime_parties(self, task_name):
        return self._task_parties[task_name]

    def get_task_node(self, task_name):
        return self._tasks[task_name]

    def get_need_revisit_tasks(self, visited_tasks, failed_tasks):
        """
        visited_tasks: already visited tasks
        failed_tasks: failed tasks

        this function finds tasks need to rerun, a task need to rerun if is upstreams is failed
        """
        invalid_tasks = set(self.topological_sort()) - set(visited_tasks)
        invalid_tasks |= set(failed_tasks)

        revisit_tasks = []
        for task_to_check in visited_tasks:
            if task_to_check in invalid_tasks:
                revisit_tasks.append(task_to_check)
                continue

            task_valid = True
            task_stack = {task_to_check}
            stack = [task_to_check]

            while len(stack) > 0 and task_valid:
                task = stack.pop()
                pre_tasks = self.predecessors(task)

                for pre_task in pre_tasks:
                    if pre_task in task_stack:
                        continue
                    if pre_task in invalid_tasks:
                        task_valid = False
                        break

                    task_stack.add(pre_task)
                    stack.append(pre_task)

            if not task_valid:
                revisit_tasks.append(task_to_check)

        return revisit_tasks

    def topological_sort(self):
        return nx.topological_sort(self._dag)

    def predecessors(self, task):
        return set(self._dag.predecessors(task))

    def successors(self, task):
        return self._dag.successors(task)

    def get_edge_attr(self, src, dst):
        return self._dag.edges[src, dst]

    @staticmethod
    def check_and_add_runtime_roles(upstream_inputs, runtime_roles):
        correct_inputs = copy.deepcopy(upstream_inputs)
        for input_key, channel_list in upstream_inputs.items():
            if isinstance(channel_list, list):
                for idx, channel in enumerate(channel_list):
                    if channel.roles is None:
                        correct_inputs[input_key][idx].roles = runtime_roles
            else:
                if channel_list.roles is None:
                    correct_inputs[input_key].roles = runtime_roles

        return correct_inputs

    @property
    def conf(self):
        return self._conf

    @classmethod
    def deploy(cls, task_name_list: list, dag_schema: DAGSchema, component_specs: Dict[str, ComponentSpec]):
        dag_parser = DagParser()
        dag_parser.parse_dag(dag_schema, component_specs)

        data_tracer = dict()
        task_name_set = set(task_name_list)
        for task_name in dag_parser.topological_sort():
            trace_task_name = cls.trace_back_deploy_task(
                task_name,
                task_name_set,
                dag_schema.dag,
                component_specs,
                data_tracer
            )
            data_tracer[task_name] = trace_task_name

        dag_spec = cls.deduce_dag(dag_parser, task_name_list, dag_schema.dag, component_specs, data_tracer)
        dag_spec = cls.erase_redundant_tasks(
            task_name_set,
            dag_spec
        )

        return dag_spec

    @classmethod
    def deduce_dag(cls, dag_parser: 'DagParser', task_name_list: list, dag_spec: DAGSpec,
                   component_specs: Dict[str, ComponentSpec], data_tracer: dict):
        stage = dag_spec.stage
        deduced_dag = copy.deepcopy(dag_spec)
        deduced_dag.stage = JobStage.PREDICT

        """
        linkage messages only occur in tasks field
        """
        task_name_set = set(task_name_list)
        topological_task_list = set(dag_parser.topological_sort())
        for task_name in topological_task_list:
            if task_name not in task_name_set:
                continue

            task = dag_spec.tasks[task_name]
            """
            job stage should be "train", so task.stage is default/predict/None, if stage is predict, erase it
            """
            if task.stage == JobStage.PREDICT:
                deduced_dag.tasks[task_name].stage = None

            task_stage = task.stage if task.stage else stage
            """
            default stage should not distinguish fit & transform or fit & transform. 
            """
            component_spec = component_specs[task_name]
            if task.inputs and task.inputs.artifacts:
                for artifact_name, artifact_channel in task.inputs.artifacts.items():
                    artifact_definition = component_spec.input_definitions.artifacts[artifact_name]
                    if artifact_definition.type in [ArtifactType.DATASET, ArtifactType.DATASETS]:
                        deduced_dag.tasks[task_name].inputs.artifacts.pop(artifact_name)
                        if artifact_name == InputDataKeyType.VALIDATE_DATA:
                            continue
                        elif artifact_name == InputDataKeyType.TRAIN_DATA:
                            """
                            change train_data to test_data, try to infer a test data key in input definition
                            """
                            test_input_key = cls.infer_test_input_data_key(artifact_name,
                                                                           task.inputs.artifacts.keys(),
                                                                           component_spec.input_definitions.artifacts)
                        else:
                            """
                            test_data/input_data or other data type, it won't be changed if stage is default or predict,
                            else also infer first
                            """
                            if JobStage.DEFAULT in artifact_definition.stages or \
                                    JobStage.PREDICT in artifact_definition.stages:
                                test_input_key = artifact_name
                            else:
                                test_input_key = cls.infer_test_input_data_key(artifact_name,
                                                                               task.inputs.artifacts.keys(),
                                                                               component_spec.input_definitions.artifacts)

                        test_input_artifact_definition = component_spec.input_definitions.artifacts[test_input_key]
                        artifact_source_type = list(artifact_channel.items())[0][0]
                        runtime_output_channel = list(artifact_channel.items())[0][1]
                        test_input_data = cls.infer_test_input_data(
                            task_name_set,
                            artifact_definition,
                            test_input_artifact_definition,
                            runtime_output_channel,
                            dag_spec,
                            component_specs,
                            data_tracer
                        )
                        if test_input_data:
                            if test_input_key not in deduced_dag.tasks[task_name].inputs.artifacts:
                                deduced_dag.tasks[task_name].inputs.artifacts[test_input_key] = dict()
                            deduced_dag.tasks[task_name].inputs.artifacts[test_input_key][artifact_source_type] = \
                                test_input_data
                    elif artifact_definition.type in [ArtifactType.MODEL, ArtifactType.MODELS]:
                        deduced_dag.tasks[task_name].inputs.artifacts.pop(artifact_name)

            model_input_artifact_key, model_output_artifact_key = cls.infer_model_artifact(
                component_spec.input_definitions.artifacts, component_spec.output_definitions.artifacts)
            if model_input_artifact_key and model_output_artifact_key:
                deduced_dag.tasks[task_name].inputs.artifacts[model_input_artifact_key] = {
                    ArtifactSourceType.MODEL_WAREHOUSE: ModelWarehouseChannelSpec(
                        producer_task=task_name,
                        output_artifact_key=model_output_artifact_key
                    )
                }

            if deduced_dag.tasks[task_name].inputs:
                deduced_dag.tasks[task_name].dependent_tasks = cls.infer_dependent_tasks(
                    deduced_dag.tasks[task_name].inputs.artifacts
                )

        return deduced_dag

    @classmethod
    def erase_redundant_tasks(cls, task_name_set: set, dag_spec: DAGSpec):
        ret_dag = copy.deepcopy(dag_spec)
        for task, task_spec in dag_spec.tasks.items():
            if task not in task_name_set:
                ret_dag.tasks.pop(task)

        if dag_spec.party_tasks:
            for site_name, party_tasks_spec in dag_spec.party_tasks.items():
                if not party_tasks_spec.tasks:
                    continue

                for task_name, task_spec in party_tasks_spec.tasks.items():
                    if task_name not in task_name_set:
                        ret_dag.party_tasks[site_name].tasks.pop(task_name)

        return ret_dag

    @classmethod
    def infer_dependent_tasks(cls, artifacts):
        if not artifacts:
            return None

        dependent_task_list = list()
        for artifact_name, artifact_channel in artifacts.items():
            for artifact_source_type, channels in artifact_channel.items():
                if artifact_source_type == ArtifactSourceType.MODEL_WAREHOUSE:
                    continue

                if not isinstance(channels, list):
                    channels = [channels]
                for channel in channels:
                    dependent_task_list.append(channel.producer_task)

        return dependent_task_list

    @classmethod
    def infer_test_input_data_key(cls, replace_key=None, train_input_keys=None, artifact_definitions=None):
        """
        we assume that training data and test_data should not be configure in same time
        """
        candidate_predict_set = set()
        candidate_default_set = set()
        candidate_training_set = set()
        for input_key, input_spec in artifact_definitions.items():
            if input_spec.type in [ArtifactType.MODEL, ArtifactType.MODELS, ArtifactType.METRIC]:
                continue
            if JobStage.DEFAULT in input_spec.stages or JobStage.PREDICT in input_spec.stages:
                if train_input_keys and input_key in train_input_keys:
                    candidate_training_set.add(input_key)
                elif JobStage.DEFAULT in input_spec.stages:
                    candidate_default_set.add(input_key)
                else:
                    candidate_predict_set.add(input_key)

        if candidate_predict_set:
            if len(candidate_predict_set) > 1:
                print("Warning: multiple test input artifact data key exists, choose one randomly")
            return list(candidate_predict_set)[0]

        if candidate_default_set:
            if len(candidate_default_set) > 1:
                print("Warning: multiple test input artifact data key exists, choose one randomly")
            return list(candidate_predict_set)[0]

        if replace_key in candidate_training_set:
            return replace_key

        if not candidate_training_set:
            raise ValueError("Can not infer test input data key")

        return list(candidate_training_set)[0]

    @classmethod
    def infer_model_artifact(cls, input_artifacts: Dict[str, ArtifactSpec], output_artifacts: Dict[str, ArtifactSpec]):
        model_input_artifact_key, model_output_artifact_key = None, None
        if output_artifacts:
            for artifact_name, artifact_spec in output_artifacts.items():
                if artifact_spec.type not in [ArtifactType.MODEL, ArtifactType.MODELS]:
                    continue

                if JobStage.TRAIN in artifact_spec.stages:
                    model_output_artifact_key = artifact_name

        if input_artifacts:
            for artifact_name, artifact_spec in input_artifacts.items():
                if artifact_spec.type not in [ArtifactType.MODEL, ArtifactType.MODELS]:
                    continue

                if JobStage.PREDICT in artifact_spec.stages or JobStage.DEFAULT in artifact_spec.stages:
                    model_input_artifact_key = artifact_name

        return model_input_artifact_key, model_output_artifact_key

    @classmethod
    def infer_test_output_data_key(cls, output_artifacts: Dict[str, ArtifactSpec]):
        candidate_artifact_name = None
        candidate_count = 0
        for artifact_name, artifact_spec in output_artifacts.items():
            if artifact_spec.type not in [ArtifactType.DATASET, ArtifactType.DATASETS]:
                continue

            if JobStage.DEFAULT not in artifact_spec.stages and JobStage.PREDICT not in artifact_spec.stages:
                continue

            if JobStage.PREDICT in artifact_spec.stages:
                return artifact_name

            """
            Component like data split, three output_data, try to infer test
            """
            if "test" in artifact_name:
                return artifact_name

            candidate_artifact_name = artifact_name
            candidate_count += 1

        if candidate_count != 1:
            raise ValueError(f"Can not infer output artifact data name from {output_artifacts}")
        return candidate_artifact_name

    @classmethod
    def infer_test_input_data(cls,
                              task_name_set,
                              train_artifact_definition,
                              test_artifact_definition,
                              output_channel: Union[RuntimeTaskOutputChannelSpec, List[RuntimeTaskOutputChannelSpec]],
                              dag_spec,
                              component_specs,
                              data_tracer: dict):
        if train_artifact_definition.type != test_artifact_definition.type:
            """
            e.g.: datasets->dataset or dataset->datasets
            """
            raise ValueError(f"train_artifact_definition's type is {train_artifact_definition.type}, "
                             f"can not be changed to {test_artifact_definition.type}")
        if isinstance(output_channel, RuntimeTaskOutputChannelSpec):
            output_channel = [output_channel]

        ret_output_channel = []
        for channel in output_channel:
            upstream_task = data_tracer[channel.producer_task]
            if upstream_task is None:
                continue

            test_artifact_data_key = cls.infer_test_output_data_key(
                component_specs[upstream_task].output_definitions.artifacts
            )
            ret_output_channel.append(
                RuntimeTaskOutputChannelSpec(
                    producer_task=upstream_task,
                    output_artifact_key=test_artifact_data_key
                )
            )

        if not ret_output_channel:
            return None
        elif train_artifact_definition.type == ArtifactType.DATASETS:
            return ret_output_channel
        else:
            return ret_output_channel[0]

    @classmethod
    def trace_back_deploy_task(cls, task_name, task_name_set, dag_spec: DAGSpec, component_specs, data_tracer: dict):
        if task_name in task_name_set:
            return task_name

        if task_name in data_tracer:
            return data_tracer[task_name]

        task_spec = dag_spec.tasks[task_name]
        component_spec = component_specs[task_name]
        if task_spec.inputs is None or task_spec.inputs.artifacts is None:
            return None

        upstream_task = set()
        for artifact_name, artifact_channel in task_spec.inputs.artifacts.items():
            artifact_definition = component_spec.input_definitions.artifacts[artifact_name]
            if artifact_definition.type not in [ArtifactType.DATASET, ArtifactType.DATASETS]:
                continue

            if not task_spec.stage:
                """
                task stage is train, inherit from job is train
                """
                if artifact_name == InputDataKeyType.VALIDATE_DATA:
                    continue

            if artifact_definition.type == ArtifactType.DATASETS:
                channels = list(artifact_channel.items())[0][1]
                for channel in channels:
                    upstream_task.add(
                        cls.trace_back_deploy_task(
                            channel.producer_task,
                            task_name_set,
                            dag_spec,
                            component_specs,
                            data_tracer)
                    )
            else:
                channel = list(artifact_channel.items())[0][1]
                upstream_task.add(
                    cls.trace_back_deploy_task(
                        channel.producer_task,
                        task_name_set,
                        dag_spec,
                        component_specs,
                        data_tracer)
                )

        if not upstream_task:
            return None
        elif len(upstream_task) == 1:
            return list(upstream_task)[0]
        else:
            return list(upstream_task)


class TaskNodeInfo(object):
    def __init__(self):
        self._runtime_parameters = None
        self._runtime_parties = None
        self._input_dependencies = None
        self._component_ref = None
        self._component_spec = None
        self._upstream_inputs = dict()
        self._stage = None
        self._conf = None

    @property
    def stage(self):
        return self._stage

    @stage.setter
    def stage(self, stage):
        self._stage = stage

    @property
    def runtime_parameters(self):
        return self._runtime_parameters

    @runtime_parameters.setter
    def runtime_parameters(self, runtime_parameters):
        self._runtime_parameters = runtime_parameters

    @property
    def runtime_parties(self):
        return self._runtime_parties

    @runtime_parties.setter
    def runtime_parties(self, runtime_parties):
        self._runtime_parties = runtime_parties

    @property
    def runtime_roles(self) -> list:
        roles = set()
        for party_spec in self._runtime_parties:
            roles.add(party_spec.role)

        return list(roles)

    @property
    def upstream_inputs(self):
        return self._upstream_inputs

    @upstream_inputs.setter
    def upstream_inputs(self, upstream_inputs):
        self._upstream_inputs = upstream_inputs

    @property
    def component_spec(self):
        return self._component_spec

    @component_spec.setter
    def component_spec(self, component_spec):
        self._component_spec = component_spec

    @property
    def output_definitions(self):
        return self._component_spec.output_definitions

    @property
    def component_ref(self):
        return self._component_ref

    @component_ref.setter
    def component_ref(self, component_ref):
        self._component_ref = component_ref

    @property
    def conf(self):
        return self._conf

    @conf.setter
    def conf(self, conf):
        self._conf = conf


class Party(BaseModel):
    role: str
    party_id: Union[str, int]
