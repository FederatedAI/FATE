from typing import List
from ...conf.types import PlaceHolder
from ..component_base import Component
from ...interface import ArtifactChannel


class HeteroLR(Component):
    yaml_define_path = "./component_define/fate/hetero_lr.yaml"

    def __init__(self,
                 name: str,
                 runtime_roles: List[str] = None,
                 learning_rate: float = PlaceHolder(),
                 max_iter: int = PlaceHolder(),
                 train_data: ArtifactChannel = PlaceHolder(),
                 validate_data: ArtifactChannel = PlaceHolder(),
                 test_data: ArtifactChannel = PlaceHolder(),
                 input_model: ArtifactChannel = PlaceHolder()
                 ):
        inputs = locals()
        self._process_init_inputs(inputs)
        super(HeteroLR, self).__init__()
        self.name = name
        self.runtime_roles = runtime_roles
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.train_data = train_data
        self.validate_data = validate_data
        self.test_data = test_data
        self.input_model = input_model
