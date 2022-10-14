from pipeline.components import Component
from pipeline.interface import Input, Output


class HeteroLR(Component):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._module = "HeteroLR"
        self.input = Input(self.name, data_key=["train_data", "validate_data", "test_data"], model_key=["model"])
        self.output = Output(self.name, output_key=["data", "model"])
