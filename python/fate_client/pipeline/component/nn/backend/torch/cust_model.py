from torch import nn
import importlib
from pipeline.component.nn.backend.torch.base import FateTorchLayer
ML_PATH = 'federatedml.nn_'

PATH = '{}.model_zoo'.format(ML_PATH)


class CustModel(FateTorchLayer, nn.Module):

    def __init__(self, name, **kwargs):
        super(CustModel, self).__init__()
        assert isinstance(name, str), 'name must be a str, specify the module file in the model_zoo'
        self.param_dict = {'name': name, 'param': kwargs}
        self._model = None

    def init_model(self):
        if self._model is None:
            self._model = self.get_pytorch_model()

    def forward(self, x):
        if self._model is None:
            raise ValueError('model not init, call init_model() function')
        return self._model(x)

    def get_pytorch_model(self):

        module_name: str = self.param_dict['name']
        module_param: dict = self.param_dict['param']
        nn_modules = importlib.import_module('{}.{}'.format(PATH, module_name))
        try:

            for k, v in nn_modules.__dict__.items():
                if isinstance(v, type):
                    if issubclass(v, nn.Module) and v is not nn.Module:
                        return v(**module_param)
            raise ValueError('Did not find any class in {}.py that is pytorch nn.Module'.
                             format(module_name))
        except ValueError as e:
            raise e

    def __repr__(self):
        return 'CustModel({})'.format(str(self.param_dict))


if __name__ == '__main__':
    pass
