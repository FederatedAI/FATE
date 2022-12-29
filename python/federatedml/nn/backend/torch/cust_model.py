import importlib

from torch import nn

from federatedml.nn.backend.torch.base import FateTorchLayer
from federatedml.nn.backend.utils.common import ML_PATH

PATH = '{}.model_zoo'.format(ML_PATH)


class CustModel(FateTorchLayer, nn.Module):

    def __init__(self, module_name, class_name, **kwargs):
        super(CustModel, self).__init__()
        assert isinstance(
            module_name, str), 'name must be a str, specify the module in the model_zoo'
        assert isinstance(
            class_name, str), 'class name must be a str, specify the class in the module'
        self.param_dict = {
            'module_name': module_name,
            'class_name': class_name,
            'param': kwargs}
        self._model = None

    def init_model(self):
        if self._model is None:
            self._model = self.get_pytorch_model()

    def forward(self, x):
        if self._model is None:
            raise ValueError('model not init, call init_model() function')
        return self._model(x)

    def get_pytorch_model(self):

        module_name: str = self.param_dict['module_name']
        class_name = self.param_dict['class_name']
        module_param: dict = self.param_dict['param']
        if module_name.endswith('.py'):
            module_name = module_name.replace('.py', '')
        nn_modules = importlib.import_module('{}.{}'.format(PATH, module_name))
        try:
            for k, v in nn_modules.__dict__.items():
                if isinstance(v, type):
                    if issubclass(
                            v, nn.Module) and v is not nn.Module and v.__name__ == class_name:
                        return v(**module_param)
            raise ValueError(
                'Did not find any class in {}.py that is pytorch nn.Module and named {}'. format(
                    module_name, class_name))
        except ValueError as e:
            raise e

    def __repr__(self):
        return 'CustModel({})'.format(str(self.param_dict))
