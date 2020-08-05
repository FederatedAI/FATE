from pipeline.component.component_base import Component
from pipeline.interface.output import Output
from federatedml.param.boosting_param import HeteroFastSecureBoostParam


class HeteroFastSecureBoost(Component, HeteroFastSecureBoostParam):
    def __init__(self, **kwargs):
        Component.__init__(self, **kwargs)

        print(self.name)
        new_kwargs = self.erase_component_base_param(**kwargs)

        HeteroFastSecureBoostParam.__init__(self, **new_kwargs)
        self.output = Output(self.name)
        self._module_name = "HeteroFastSecureBoost"