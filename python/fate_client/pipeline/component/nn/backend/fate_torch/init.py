import copy
from torch.nn import init as torch_init
import functools
from pipeline.component.nn.backend.fate_torch.base import FateTorchLayer
from pipeline.component.nn.backend.fate_torch.base import Sequential

str_init_func_map = {
    "uniform": torch_init.uniform_,
    "normal": torch_init.normal_,
    "constant": torch_init.constant_,
    "xavier_uniform": torch_init.xavier_uniform_,
    "xavier_normal": torch_init.xavier_normal_,
    "kaiming_uniform": torch_init.kaiming_uniform_,
    "kaiming_normal": torch_init.kaiming_normal_,
    "eye": torch_init.eye_,
    "dirac": torch_init.dirac_,
    "orthogonal": torch_init.orthogonal_,
    "sparse": torch_init.sparse_,
    "zeros": torch_init.zeros_,
    "ones": torch_init.ones_
}


#
# def extract_param(func):
#
#     args = inspect.getargspec(func)
#     keys = args[0][1:]
#     if len(keys) == 0:
#         return {}
#     defaults = args[-1]
#     args_map = {}
#     if defaults is not None:
#         for idx, i in enumerate(keys[-len(defaults):]):
#             args_map[i] = defaults[idx]
#
#     for i in keys:
#         if i not in args_map:
#             args_map[i] = Required()
#
#     return args_map


def init_weight(m, initializer):
    if hasattr(m, 'weight'):
        initializer(m.weight)
    # LSTM RNN
    if hasattr(m, 'weight_hh_l0'):
        initializer(m.weight_hh_l0)
    # LSTM RNN
    if hasattr(m, 'weight_ih_l0'):
        initializer(m.weight_ih_l0)


def init_bias(m, initializer):
    if hasattr(m, 'bias') and not isinstance(m.bias, bool):  # LSTM, RNN .bias is bool
        initializer(m.bias)
    # LSTM RNN
    if hasattr(m, 'bias_hh_l0'):
        initializer(m.bias_hh_l0)
    # LSTM RNN
    if hasattr(m, 'bias_ih_l0'):
        initializer(m.bias_ih_l0)


def get_init_func_type(init='weight'):
    if init == 'weight':
        return init_weight
    elif init == 'bias':
        return init_bias
    else:
        return None


def recursive_init(m, init_func, obj):
    if len(list(m.children())) > 0:
        if m == obj:
            return
        recursive_init(m, init_func, m)
    else:
        try:
            init_func(m)
        except Exception as e:
            print('initialize layer {} failed, exception is :{}'.format(m, e))


def make_apply_func(torch_initializer, param_dict, init_func, layer):
    param_dict.pop('layer')
    param_dict.pop('init')
    initializer = functools.partial(torch_initializer, **param_dict)
    init_func = functools.partial(init_func, initializer=initializer)
    recursive_init_func = functools.partial(recursive_init, obj=layer, init_func=init_func)
    return recursive_init_func, param_dict


def get_init_dict(init_func, param_dict, init_type):
    rev_dict = {v: k for k, v in str_init_func_map.items()}
    rs = {'init_type': init_type, 'init_func': rev_dict[init_func], 'param': param_dict}
    return rs


def record_initializer(layers: FateTorchLayer, init_dict):
    if init_dict['init_type'] == 'weight':
        layers.initializer['weight'] = init_dict
    elif init_dict['init_type'] == 'bias':
        layers.initializer['bias'] = init_dict


def run_init(torch_initializer, input_var, init, layer):
    if isinstance(layer, Sequential):
        for sub_layer in layer:
            run_init(torch_initializer, input_var, init, sub_layer)
    elif isinstance(layer, FateTorchLayer):
        recursive_init_func, param_dict = make_apply_func(torch_initializer, copy.deepcopy(input_var),
                                                          get_init_func_type(init), layer)
        layer.apply(recursive_init_func)
        record_initializer(layer, get_init_dict(torch_initializer, param_dict, init))
    else:
        try:
            return torch_initializer(layer)
        except Exception:
            pass


"""
Init Func
"""


def uniform_(layer, a=0, b=1, init='weight'):
    run_init(str_init_func_map['uniform'], copy.deepcopy(locals()), init, layer)


def normal_(layer, mean=0, std=1, init='weight'):
    run_init(str_init_func_map['normal'], copy.deepcopy(locals()), init, layer)


def constant_(layer, val, init='weight'):
    run_init(str_init_func_map['constant'], copy.deepcopy(locals()), init, layer)


def ones_(layer, init='weight'):
    run_init(str_init_func_map['ones'], copy.deepcopy(locals()), init, layer)


def zeros_(layer, init='weight'):
    run_init(str_init_func_map['zeros'], copy.deepcopy(locals()), init, layer)


def eye_(layer, init='weight'):
    run_init(str_init_func_map['eye'], copy.deepcopy(locals()), init, layer)


def dirac_(layer, group=1, init='weight'):
    run_init(str_init_func_map['dirac'], copy.deepcopy(locals()), init, layer)


def xavier_uniform_(layer, gain=1.0, init='weight'):
    run_init(str_init_func_map['xavier_uniform'], copy.deepcopy(locals()), init, layer)


def xavier_normal_(layer, gain=1.0, init='weight'):
    run_init(str_init_func_map['xavier_normal'], copy.deepcopy(locals()), init, layer)


def kaiming_uniform_(layer, a=0, mode='fan_in', nonlinearity='leaky_relu', init='weight'):
    run_init(str_init_func_map['kaiming_uniform'], copy.deepcopy(locals()), init, layer)


def kaiming_normal_(layer, a=0, mode='fan_in', nonlinearity='leaky_relu', init='weight'):
    run_init(str_init_func_map['kaiming_normal'], copy.deepcopy(locals()), init, layer)


def orthogonal_(layer, gain=1, init='weight'):
    run_init(str_init_func_map['orthogonal'], copy.deepcopy(locals()), init, layer)


def sparse_(layer, sparsity, std=0.01, init='weight'):
    run_init(str_init_func_map['sparse'], copy.deepcopy(locals()), init, layer)


str_fate_torch_init_func_map = {
    "uniform": uniform_,
    "normal": normal_,
    "constant": constant_,
    "xavier_uniform": xavier_uniform_,
    "xavier_normal": xavier_normal_,
    "kaiming_uniform": kaiming_uniform_,
    "kaiming_normal": kaiming_normal_,
    "eye": eye_,
    "dirac": dirac_,
    "orthogonal": orthogonal_,
    "sparse": sparse_,
    "zeros": zeros_
}

if __name__ == '__main__':
    pass
