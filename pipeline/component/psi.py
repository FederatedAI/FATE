from pipeline.component.component_base import Component
from pipeline.interface.output import Output
from federatedml.param.psi_param import PSIParam


class PSI(Component, PSIParam):
    def __init__(self, **kwargs):
        Component.__init__(self, **kwargs)

        print(self.name)
        new_kwargs = self.erase_component_base_param(**kwargs)

        PSIParam.__init__(self, **new_kwargs)
        self.output = Output(self.name, has_model=False)
        self._module_name = "PSI"