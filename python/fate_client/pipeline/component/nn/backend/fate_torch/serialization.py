import copy
import inspect
from collections import OrderedDict
try:
    from pipeline.component.nn.backend.fate_torch import optim, init, nn
    from pipeline.component.nn.backend.fate_torch import operation
    from pipeline.component.nn.backend.fate_torch.base import Sequential
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

    nn_layer_class = nn_dict[class_name]
    # init_para_key = inspect.getfullargspec(nn_layer_class.__init__)[0]
    # init_para_key.remove('self')
    # param_dict = {}
    # for k in init_para_key:
    #     if k in nn_define:
    #         param_dict[k] = nn_define[k]

    layer = nn_layer_class(**init_param_dict)

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

    return Sequential(add_dict)


def recover_optimizer_from_dict(define_dict):
    opt_dict = dict(inspect.getmembers(optim))
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
    opt_dict = {
        "lr": 0.01,
        "betas": [
            0.9,
            0.999
        ],
        "eps": 1e-08,
        "weight_decay": 0,
        "amsgrad": False,
        "optimizer": "Adam"
    }
    optimizer = recover_optimizer_from_dict(opt_dict)

    loss_fn_define = {
        "weight": None,
        "size_average": None,
        "reduce": None,
        "reduction": "mean",
        "loss_fn": "BCELoss"
    }
    loss_fn = recover_loss_fn_from_dict(loss_fn_define)

    test = {'0-0': {'bias': True, 'in_features': 8, 'initializer': {}, 'layer': 'Linear', 'out_features': 4}}
    print(modify_linear_input_shape(16, test))
