import copy
import networkx as nx

from pydantic import BaseModel
from typing import Dict, Union

from ..entity.dag_structures import DAGSchema
from ..entity.component_structures import ComponentSpec


class DagParser(object):
    def __init__(self):
        self._dag = nx.DiGraph()
        self._links = dict()
        self._task_parameters = dict()
        self._task_parties = dict()
        self._tasks = dict()

    def parse_dag(self, dag_schema: DAGSchema, component_specs: Dict[str, ComponentSpec] = None):
        dag_spec = dag_schema.dag
        dag_stage = dag_spec.stage
        tasks = dag_spec.tasks
        for name, task_spec in tasks.items():
            task_stage = dag_stage
            if task_spec.stage:
                task_stage = task_spec.stage

            self._tasks[name] = TaskNodeInfo()
            self._tasks[name].stage = task_stage
            if component_specs:
                self._tasks[name].component_spec = component_specs[name]
            self._init_task_runtime_parameters(name, dag_schema)

            if not task_spec.inputs or not task_spec.inputs.artifacts:
                continue

            upstream_inputs = dict()
            for input_key, output_specs_dict in task_spec.inputs.artifacts.items():
                upstream_inputs[input_key] = dict()
                for output_name, channel_spec_list in output_specs_dict.items():
                    upstream_inputs[input_key] = channel_spec_list
                    if not isinstance(channel_spec_list, list):
                        channel_spec_list = [channel_spec_list]

                    for channel_spec in channel_spec_list:
                        dependent_task = channel_spec.producer_task
                        self._add_edge(dependent_task, name)

            self._tasks[name].upstream_inputs = upstream_inputs

    def _add_edge(self, src, dst, attrs=None):
        if not attrs:
            attrs = {}

        self._dag.add_edge(src, dst, **attrs)

    def _init_task_runtime_parameters(self, task_name: str, dag_schema: DAGSchema):
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
        task_runtime_parties = []

        for party in dag.parties:
            if party.role not in role_keys:
                continue
            task_parameters[party.role] = dict()
            for party_id in party.party_id:
                task_parameters[party.role][party_id] = copy.deepcopy(common_parameters)
                task_runtime_parties.append(Party(role=party.role, party_id=party_id))

        if dag.party_tasks:
            party_tasks = dag.party_tasks
            for site_name, party_tasks_spec in party_tasks.items():
                if task_name not in party_tasks_spec.tasks:
                    continue
                party_parties = party_tasks_spec.parties
                party_task_spec = party_tasks_spec.tasks[task_name]
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

    def get_runtime_parties(self, task_name):
        return self._task_parties[task_name]

    def get_task_node(self, task_name):
        return self._tasks[task_name]

    def topological_sort(self):
        return nx.topological_sort(self._dag)

    def predecessors(self, node):
        return set(self._dag.predecessors(node))

    def successors(self, node):
        return self._dag.successors(node)

    def get_edge_attr(self, src, dst):
        return self._dag.edges[src, dst]


class TaskNodeInfo(object):
    def __init__(self):
        self._runtime_parameters = None
        self._runtime_parties = None
        self._input_dependencies = None
        self._component_spec = None
        self._upstream_inputs = dict()
        self._stage = None

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
        return self._component_spec.name


class Party(BaseModel):
    role: str
    party_id: Union[str, int]
