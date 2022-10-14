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

        self._is_compiled = False

    def add_node(self, node, **kwargs):
        if node.name in self._node_def:
            raise ValueError(f"{node.name} has been added before")

        self._node_def[node.name] = node
        self._dag.add_node(node.name)

        data_key = node.input.get_data_key(default={})
        for key in data_key:
            if kwargs.get(key) is not None:
                self._links[LinkKey.DATA][node.name] = kwargs.pop(key)

        model_key = node.input.get_model_key(default={})
        for key in model_key:
            if kwargs.get(key) is not None:
                self._links[LinkKey.MODEL][node.name] = kwargs.pop(key)

        cache_key = node.input.get_model_key(default={})
        for key in cache_key:
            if kwargs.get(key) is not None:
                self._links[LinkKey.CACHE][node.name] = kwargs.pop(key)

    def add_edge(self, src, dst, attrs=None):
        if not attrs:
            attrs = {}

        self._dag.add_edge(src, dst, attrs)

    def get_node(self, src):
        return self._node_def[src]

    def compile(self):
        """
        add edge after compiled
        """
        for link_outer_key, links in self._links.items():
            for dst, attrs in links.items():
                for data_key, sources in links.items():
                    if isinstance(sources, str):
                        sources = [sources]

                    for src in sources:
                        self.add_edge(src, dst, attrs=dict(
                            link_outer_key={data_key}
                        ))

        self._is_compiled = True

    def topological_sort(self):
        return self._dag.topological_sort()

    def predecessors(self, node):
        return self._dag.predecessors(node)

    def successors(self, node):
        return self._dag.successors(node)

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

    def get_get_executable_node_info(self, node):
        ...


