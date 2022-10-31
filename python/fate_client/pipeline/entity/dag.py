import networkx as nx
from ..conf.types import LinkKey


class DAG(object):
    def __init__(self):
        self._dag = nx.DiGraph()
        self._node_def = dict()

        self._node_local_inputs = dict()
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

        node_input_interface = node.get_input_interface()

        for inner_keys, outer_key in node_input_interface:
            for inner_key in inner_keys:
                if kwargs.get(inner_key) is None:
                    continue
                if outer_key not in self._links or inner_key not in self._links[outer_key]:
                    self._links[outer_key] = {
                        inner_key: dict()
                    }

                if node.name not in self._links[outer_key][inner_key]:
                    self._links[outer_key][inner_key][node.name] = list()

                self._links[outer_key][inner_key][node.name].append(kwargs.pop(inner_key))

    def add_edge(self, src, dst, attrs=None):
        if not attrs:
            attrs = {}

        self._dag.add_edge(src, dst, **attrs)

    def get_node(self, src):
        return self._node_def[src]

    def get_runtime_parties(self, node_name):
        node = self._node_def[node_name]
        runtime_roles = set(node.get_support_roles()) & set(self._roles.get_runtime_roles())
        parties = dict()
        for role in runtime_roles:
            role_party_list = self._roles.get_party_list_by_role(role)
            parties[role] = role_party_list

        return parties

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

    def get_module(self, node_name):
        return self._node_def[node_name].module

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
            for inner_key, inner_links in links.items():
                for dst, sources in inner_links.items():
                    for src in sources:
                        # TODO: this should be optimize, local input maybe a LocalInputComponent
                        src_node_name, src_link = src.split(".", 1)
                        self.add_edge(src_node_name, dst, attrs={
                            link_outer_key: {
                                inner_key: src
                            }
                        })

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

    def get_node_output_interface(self, node_name):
        node = self._node_def[node_name]
        return node.get_output_interface()

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
