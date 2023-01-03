from typing import List
from ...conf.types import PlaceHolder
from ..component_base import Component
from ...interface import ArtifactChannel


class Evaluation(Component):
    yaml_define_path = "./component_define/fate/evaluation.yaml"

    def __init__(self,
                 name: str,
                 runtime_roles: List[str] = None,
                 input_data: ArtifactChannel = PlaceHolder(),
                 eval_type: str = PlaceHolder(),
                 ):
        inputs = locals()
        self._process_init_inputs(inputs)
        super(Evaluation, self).__init__()
        self.name = name
        self.runtime_roles = runtime_roles
        self.input_data = input_data
        self.eval_type = eval_type
