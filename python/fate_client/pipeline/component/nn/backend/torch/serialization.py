import copy
import inspect
from collections import OrderedDict
try:
    from torch.nn import Sequential as tSeq
    from pipeline.component.nn.backend.torch import optim, init, nn
    from pipeline.component.nn.backend.torch import operation
    from pipeline.component.nn.backend.torch.base import Sequential, get_torch_instance
    from pipeline.component.nn.backend.torch.cust import CustModel, CustLoss
    from pipeline.component.nn.backend.torch.interactive import InteractiveLayer
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
        raise ValueError(
            'no layer or operation info found in nn define, please check your layer config and make'
            'sure they are correct for pytorch backend')

    if 'initializer' in init_param_dict:
        init_param_dict.pop('initializer')

    # find corresponding class
    if class_name == CustModel.__name__:
        nn_layer_class = CustModel
    elif class_name == InteractiveLayer.__name__:
        nn_layer_class = InteractiveLayer
    else:
        nn_layer_class = nn_dict[class_name]

    # create layer or Module
    if nn_layer_class == CustModel:  # converto to pytorch model
        layer: CustModel = CustModel(module_name=init_param_dict['module_name'],
                                     class_name=init_param_dict['class_name'],
                                     **init_param_dict['param'])
        layer = layer.get_pytorch_model()
    elif nn_layer_class == InteractiveLayer:
        layer: InteractiveLayer = InteractiveLayer(**init_param_dict)
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

    return layer, class_name


def recover_sequential_from_dict(nn_define):
    nn_define_dict = nn_define
    nn_dict = dict(inspect.getmembers(nn))
    op_dict = dict(inspect.getmembers(operation))
    nn_dict.update(op_dict)

    class_name_list = []
    try:
        # submitted model have int prefixes, they make sure that layers are in
        # order
        add_dict = OrderedDict()
        keys = list(nn_define_dict.keys())
        keys = sorted(keys, key=lambda x: int(x.split('-')[0]))
        for k in keys:
            layer, class_name = recover_layer_from_dict(nn_define_dict[k], nn_dict)
            add_dict[k] = layer
            class_name_list.append(class_name)
    except BaseException:
        add_dict = OrderedDict()
        for k, v in nn_define_dict.items():
            layer, class_name = recover_layer_from_dict(v, nn_dict)
            add_dict[k] = layer
            class_name_list.append(class_name)

    if len(class_name_list) == 1 and class_name_list[0] == CustModel.__name__:
        # If there are only a CustModel, return the model only
        return list(add_dict.values())[0]
    else:
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
    if define_dict['loss_fn'] == CustLoss.__name__:
        return CustLoss(loss_module_name=param_dict['loss_module_name'],
                        class_name=param_dict['class_name'],
                        **param_dict['param']).get_pytorch_model()
    else:
        return loss_fn_dict[define_dict['loss_fn']](**param_dict)


if __name__ == '__main__':
    pass
