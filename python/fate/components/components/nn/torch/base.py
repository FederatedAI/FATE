import torch as t
from fate.components.components.nn.loader import Loader
from torch.nn import Sequential as tSequential
import json


def convert_tuples_to_lists(data):
    if isinstance(data, tuple):
        return list(data)
    elif isinstance(data, list):
        return [convert_tuples_to_lists(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_tuples_to_lists(
            value) for key, value in data.items()}
    else:
        return data


class TorchModule(object):

    def __init__(self):
        t.nn.Module.__init__(self)
        self.param_dict = dict()
        self.optimizer = None

    def to_dict(self):
        ret_dict = {
            'module_name': 'torch.nn',
            'item_name': str(type(self).__name__),
            'kwargs': convert_tuples_to_lists(self.param_dict)
        }
        return ret_dict


class TorchOptimizer(object):

    def __init__(self):
        self.param_dict = dict()
        self.torch_class = None

    def to_dict(self):
        ret_dict = {
            'module_name': 'torch.optim',
            'item_name': type(self).__name__,
            'kwargs': convert_tuples_to_lists(self.param_dict)
        }
        return ret_dict

    def check_params(self, params):

        if isinstance(
                params,
                TorchModule) or isinstance(
                params,
                Sequential):
            params.add_optimizer(self)
            params = params.parameters()
        else:
            params = params

        l_param = list(params)
        if len(l_param) == 0:
            # fake parameters, for the case that there are only cust model
            return [t.nn.Parameter(t.Tensor([0]))]

        return l_param

    def register_optimizer(self, input_):

        if input_ is None:
            return
        if isinstance(
                input_,
                TorchModule) or isinstance(
                input_,
                Sequential):
            input_.add_optimizer(self)

    def to_torch_instance(self, parameters):
        return self.torch_class(parameters, **self.param_dict)


def load_seq(seq_conf: dict) -> None:

    confs = list(dict(sorted(seq_conf.items())).values())
    model_list = []
    for conf in confs:
        layer = Loader.from_dict(conf)()
        model_list.append(layer)

    return tSequential(*model_list)


class Sequential(tSequential):

    def to_dict(self):
        """
        get the structure of current sequential
        """
        layer_confs = {}
        idx = 0
        for k in self._modules:
            ordered_name = idx
            layer_confs[ordered_name] = self._modules[k].to_dict()
            idx += 1
        ret_dict = {
            'module_name': 'fate.components.components.nn.torch.base',
            'item_name': load_seq.__name__,
            'kwargs': {'seq_conf': layer_confs}
        }
        return ret_dict

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)
