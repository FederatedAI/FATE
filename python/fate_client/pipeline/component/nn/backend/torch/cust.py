from torch import nn
import importlib
from pipeline.component.nn.backend.torch.base import FateTorchLayer, FateTorchLoss
import difflib


MODEL_PATH = None
LOSS_PATH = None


def str_simi(str_a, str_b):
    return difflib.SequenceMatcher(None, str_a, str_b).quick_ratio()


def get_class(module_name, class_name, param, base_path):

    if module_name.endswith('.py'):
        module_name = module_name.replace('.py', '')
    nn_modules = importlib.import_module('{}.{}'.format(base_path, module_name))
    try:
        name_simi_list = []
        for k, v in nn_modules.__dict__.items():
            if isinstance(v, type):
                if issubclass(v, nn.Module) and v is not nn.Module:
                    if v.__name__ == class_name:
                        return v(**param)
                    else:
                        name_simi_list += ([(str_simi(class_name, v.__name__), v)])

        sort_by_simi = sorted(name_simi_list, key=lambda x: -x[0])

        if len(sort_by_simi) > 0:
            raise ValueError(
                'Did not find any class in {}.py that is subclass of nn.Module and named {}. Do you mean {}?'. format(
                    module_name, class_name, sort_by_simi[0][1].__name__))
        else:
            raise ValueError('Did not find any class in {}.py that is subclass of nn.Module and named {}'.
                             format(module_name, class_name))

    except ValueError as e:
        raise e


class CustModel(FateTorchLayer, nn.Module):

    def __init__(self, module_name, class_name, **kwargs):
        super(CustModel, self).__init__()
        assert isinstance(module_name, str), 'name must be a str, specify the module in the model_zoo'
        assert isinstance(class_name, str), 'class name must be a str, specify the class in the module'
        self.param_dict = {'module_name': module_name, 'class_name': class_name, 'param': kwargs}
        self._model = None

    def init_model(self):
        if self._model is None:
            self._model = self.get_pytorch_model()

    def forward(self, x):
        if self._model is None:
            raise ValueError('model not init, call init_model() function')
        return self._model(x)

    def get_pytorch_model(self, module_path=None):

        if module_path is None:
            return get_class(
                self.param_dict['module_name'],
                self.param_dict['class_name'],
                self.param_dict['param'],
                MODEL_PATH)
        else:
            return get_class(
                self.param_dict['module_name'],
                self.param_dict['class_name'],
                self.param_dict['param'],
                module_path)

    def __repr__(self):
        return 'CustModel({})'.format(str(self.param_dict))


class CustLoss(FateTorchLoss, nn.Module):

    def __init__(self, loss_module_name, class_name, **kwargs):
        super(CustLoss, self).__init__()
        assert isinstance(loss_module_name, str), 'loss module name must be a str, specify the module in the model_zoo'
        assert isinstance(class_name, str), 'class name must be a str, specify the class in the module'
        self.param_dict = {'loss_module_name': loss_module_name, 'class_name': class_name, 'param': kwargs}
        self._loss_fn = None

    def init_loss_fn(self):
        if self._loss_fn is None:
            self._loss_fn = self.get_pytorch_model()

    def forward(self, pred, label):
        if self._loss_fn is None:
            raise ValueError('loss not init, call init_loss_fn() function')
        return self._loss_fn(pred, label)

    def get_pytorch_model(self, module_path=None):

        module_name: str = self.param_dict['loss_module_name']
        class_name: str = self.param_dict['class_name']
        module_param: dict = self.param_dict['param']
        if module_path is None:
            return get_class(module_name=module_name, class_name=class_name, param=module_param, base_path=LOSS_PATH)
        else:
            return get_class(module_name=module_name, class_name=class_name, param=module_param, base_path=module_path)

    def __repr__(self):
        return 'CustLoss({})'.format(str(self.param_dict))
