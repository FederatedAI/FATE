import networkx as nx
from ..conf.types import LinkKey


class DAG(object):
    def __init__(self):
        self._dag = nx.DiGraph()
        self._node_def = dict()

        self._links = dict()

        self._links[LinkKey.DATA] = dict()
        self._links[LinkKey.MODEL] = dict()
        self._links[LinkKey.CACHE] = dict()

        self._roles = None
        self._is_compiled = False

    def add_node(self, node, **kwargs):
        if node.name in self._node_def:
            raise ValueError(f"{node.name} has been added before")

        self._node_def[node.name] = node
        self._dag.add_node(node.name)

        data_key = node.input.get_input_key(key="data")
        for key in data_key:
            if kwargs.get(key) is not None:
                self._links[LinkKey.DATA][node.name] = kwargs.pop(key)

        model_key = node.input.get_input_key(key="model")
        for key in model_key:
            if kwargs.get(key) is not None:
                self._links[LinkKey.MODEL][node.name] = kwargs.pop(key)

        cache_key = node.input.get_input_key(key="cache")
        for key in cache_key:
            if kwargs.get(key) is not None:
                self._links[LinkKey.CACHE][node.name] = kwargs.pop(key)

    def add_edge(self, src, dst, attrs=None):
        if not attrs:
            attrs = {}

        self._dag.add_edge(src, dst, attrs)

    def get_node(self, src):
        return self._node_def[src]

    def get_node_conf(self, node_name):
        node = self._node_def[node_name]
        runtime_roles = set(node.get_support_roles()) & set(self._roles.get_runtime_roles())
        node_conf = dict()
        if not runtime_roles:
            raise ValueError(f"{node_name} can not be executed, its support roles is {node.get_support_roles()}, "
                             f"but pipeline's runtime roles is {self._roles.get_runtime_roles}, have a check!")

        for role in runtime_roles:
            node_conf[role] = dict()
            role_party_list = self._roles.get_party_list_by_role(role)
            for idx, party_id in enumerate(role_party_list):
                conf = node.get_role_param(role, idx)
                node_conf[role][party_id] = conf

        return node_conf

    def set_roles(self, roles):
        self._roles = roles

    @property
    def leader_role(self):
        return self._roles.leader

    def compile(self):
        """
        add edge after compiled
        """
        for link_outer_key, links in self._links.items():
            for dst, attrs in links.items():
                for key, sources in links.items():
                    if isinstance(sources, str):
                        sources = [sources]

                    for src in sources:
                        self.add_edge(src, dst, attrs=dict(
                            link_outer_key={key}
                        ))

        for node_name, node in self._node_def.items():
            node.validate_runtime_env(self._roles)

        self._is_compiled = True

    def topological_sort(self):
        return nx.topological_sort(self._dag)

    def predecessors(self, node):
        return set(self._dag.predecessors(node))

    def successors(self, node):
        return self._dag.successors(node)

    def get_edge_attr(self, src, dst):
        return self._dag.edges[src, dst]

    def display(self):
        if not self._is_compiled:
            self.compile()

            
class FateFlowDAG(DAG):
    def __init__(self):
        super(FateFlowDAG, self).__init__()
        self._dsl = dict()
        self._conf = dict()

    def compile(self):
        super(FateFlowDAG, self).compile()

        self._construct_dsl_and_conf()

    def _construct_dsl_and_conf(self):
        ...


class FateStandaloneDAG(DAG):
    def __init__(self):
        super(FateStandaloneDAG, self).__init__()
