from typing import List
from ...conf.types import PlaceHolder
from ..component_base import Component
from ...interface import ArtifactChannel


class FeatureScale(Component):
    yaml_define_path = "./component_define/fate/feature_scale.yaml"

    def __init__(self,
                 name: str,
                 runtime_roles: List[str] = None,
                 method: str = PlaceHolder(),
                 train_data: ArtifactChannel = PlaceHolder(),
                 test_data: ArtifactChannel = PlaceHolder(),
                 input_model: ArtifactChannel = PlaceHolder()
                 ):
        inputs = locals()
        self._process_init_inputs(inputs)
        super(FeatureScale, self).__init__()
        self.name = name
        self.runtime_roles = runtime_roles
        self.method = method
        self.train_data = train_data
        self.test_data = test_data
        self.input_model = input_model
