from ..component_base import Component
from ...interface import Input, Output
from ...conf.types import SupportRole


class HeteroLR(Component):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._module = "HeteroLR"
        self.input = Input(self.name, data_key=["train_data", "validate_data", "test_data"], model_key=["model"])
        self.output = Output(self.name, data_key=["data"], model_key=["model"])
        self._support_roles = [SupportRole.GUEST, SupportRole.HOST, SupportRole.ARBITER]
