from typing import List
from ...conf.types import PlaceHolder
from ..component_base import Component


class Reader(Component):
    yaml_define_path = "./component_define/fate/reader.yaml"

    def __init__(self,
                 name: str,
                 runtime_roles: List[str] = None,
                 path: str = PlaceHolder()
                 ):
        inputs = locals()
        self._process_init_inputs(inputs)
        super(Reader, self).__init__()
        self.name = name
        self.runtime_roles = runtime_roles
        self.path = path
