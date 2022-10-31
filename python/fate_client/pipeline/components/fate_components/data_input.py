from ..component_base import Component
from ...interface import Input, Output
from ...conf.types import SupportRole


class DataInput(Component):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._module = "data_input"
        self.input = Input(self.name)
        self.output = Output(self.name, data_key=["data"])
        self._support_roles = [SupportRole.GUEST, SupportRole.HOST, SupportRole.ARBITER]
