import copy
import inspect
from collections import OrderedDict
try:
    from torch.nn import Sequential as tSeq
    from federatedml.nn_.backend.torch import optim, init, nn
    from federatedml.nn_.backend.torch import operation
    from federatedml.nn_.backend.torch.base import Sequential, get_torch_instance
    from federatedml.nn_.backend.torch.cust_model import CustModel
except ImportError:
    pass


def recover_layer_from_dict(nn_define, nn_dict):

    init_param_dict = copy.deepcopy(nn_define)
    if 'layer' in nn_define:
        class_name = nn_define['layer']
        init_param_dict.pop('layer')
    elif 'op' in nn_define:
        class_name = nn_define['op']
        init_param_dict.pop('op')
    else:
        raise ValueError('no layer or operation info found in nn define, please check your layer config and make'
                         'sure they are correct for pytorch backend')

    if 'initializer' in init_param_dict:
        init_param_dict.pop('initializer')

    # find corresponding class
    if class_name == CustModel.__name__:
        nn_layer_class = CustModel
    else:
        nn_layer_class = nn_dict[class_name]

    # create layer or Module
    if nn_layer_class == CustModel:  # converto to pytorch model
        layer: CustModel = CustModel(name=init_param_dict['name'], **init_param_dict['param'])
        layer = layer.get_pytorch_model()
    else:
        layer = get_torch_instance(nn_layer_class, init_param_dict)

    # initialize if there are configs
    if 'initializer' in nn_define:
        if 'weight' in nn_define['initializer']:
            init_para = nn_define['initializer']['weight']
            init_func = init.str_fate_torch_init_func_map[init_para['init_func']]
            init_func(layer, **init_para['param'])

        if 'bias' in nn_define['initializer']:
            init_para = nn_define['initializer']['bias']
            init_func = init.str_fate_torch_init_func_map[init_para['init_func']]
            init_func(layer, init='bias', **init_para['param'])

    return layer


def recover_sequential_from_dict(nn_define):
    nn_define_dict = nn_define
    add_dict = OrderedDict()
    nn_dict = dict(inspect.getmembers(nn))
    op_dict = dict(inspect.getmembers(operation))
    nn_dict.update(op_dict)
    for k, v in nn_define_dict.items():
        layer = recover_layer_from_dict(v, nn_dict)
        add_dict[k] = layer

    return tSeq(add_dict)


def recover_optimizer_from_dict(define_dict):
    opt_dict = dict(inspect.getmembers(optim))
    from federatedml.util import LOGGER
    LOGGER.debug('define dict is {}'.format(define_dict))
    if 'optimizer' not in define_dict:
        raise ValueError('please specify optimizer type in the json config')
    opt_class = opt_dict[define_dict['optimizer']]
    param_dict = copy.deepcopy(define_dict)
    if 'optimizer' in param_dict:
        param_dict.pop('optimizer')
    if 'config_type' in param_dict:
        param_dict.pop('config_type')
    return opt_class(**param_dict)


def recover_loss_fn_from_dict(define_dict):
    loss_fn_dict = dict(inspect.getmembers(nn))
    if 'loss_fn' not in define_dict:
        raise ValueError('please specify loss function in the json config')
    param_dict = copy.deepcopy(define_dict)
    param_dict.pop('loss_fn')
    return loss_fn_dict[define_dict['loss_fn']](**param_dict)


if __name__ == '__main__':
    pass
