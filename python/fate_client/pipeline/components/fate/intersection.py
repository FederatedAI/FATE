from typing import List
from ...conf.types import PlaceHolder
from ..component_base import Component
from ...interface import ArtifactChannel


class Intersection(Component):
    yaml_define_path = "./component_define/fate/intersection.yaml"

    def __init__(self,
                 name: str,
                 runtime_roles: List[str] = None,
                 input_data: ArtifactChannel = PlaceHolder(),
                 method: str = PlaceHolder(),
                 ):
        inputs = locals()
        self._process_init_inputs(inputs)
        super(Intersection, self).__init__()
        self.name = name
        self.runtime_roles = runtime_roles
        self.input_data = input_data
        self.method = method
