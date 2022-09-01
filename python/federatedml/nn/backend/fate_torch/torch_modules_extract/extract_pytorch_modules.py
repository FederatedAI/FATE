import inspect
from torch.nn.modules import linear, activation, rnn, dropout, sparse, pooling, conv, transformer, batchnorm
from torch.nn.modules import padding, pixelshuffle
from torch.nn.modules import loss


class Required(object):

    def __init__(self):
        pass

    def __repr__(self):
        return '(Required Parameter)'


def get_all_class_obj(module, key_word=''):
    members = inspect.getmembers(module)
    rs = []
    module_name = None
    for name, obj in members:
        if inspect.isclass(obj):
            if 'modules.' + key_word in obj.__module__:
                rs.append(obj)
                # print(obj)
                module_name = obj.__module__.split('.')[-1]

    return rs, module_name


def extract_init_param(class_):
    args = inspect.getargspec(class_.__init__)
    keys = args[0][1:]
    if len(keys) == 0:
        return {}
    defaults = args[-1]
    args_map = {}
    if defaults is not None:
        for idx, i in enumerate(keys[-len(defaults):]):
            args_map[i] = defaults[idx]

    for i in keys:
        if i not in args_map:
            args_map[i] = Required()

    return args_map


def code_assembly(param, nn_class, module_name):
    if module_name == 'loss':
        parent_class = 'FateTorchLoss'
    else:
        parent_class = 'FateTorchLayer'

    para_str = ""
    non_default_param = ""
    init_str = """"""
    for k, v in param.items():

        new_para = "\n        self.param_dict['{}'] = {}".format(k, k)
        init_str += new_para
        if isinstance(v, Required):
            non_default_param += str(k)
            non_default_param += ', '
            continue

        para_str += str(k)
        if isinstance(v, str):
            para_str += "='{}'".format(v)
        else:
            para_str += "={}".format(str(v))
        para_str += ', '

    para_str = non_default_param + para_str

    init_ = """
    def __init__(self, {}**kwargs):
        {}.__init__(self){}
        self.param_dict.update(kwargs)
        nn.modules.{}.{}.__init__(self, **self.param_dict)
    """.format(para_str, parent_class, init_str, module_name, nn_class)

    code = """
class {}({}, {}):
        {}
    """.format(nn_class, 'nn.modules.{}.{}'.format(module_name, nn_class), parent_class, init_)

    return code


if __name__ == '__main__':

    rs1 = get_all_class_obj(linear, 'linear')
    rs2 = get_all_class_obj(rnn, 'rnn')
    rs3 = get_all_class_obj(sparse, 'sparse')
    rs4 = get_all_class_obj(dropout, 'dropout')
    rs5 = get_all_class_obj(activation, 'activation')
    rs6 = get_all_class_obj(conv, 'conv')
    rs7 = get_all_class_obj(transformer, 'transformer')
    rs8 = get_all_class_obj(pooling, 'pooling')
    rs9 = get_all_class_obj(batchnorm, 'batchnorm')
    rs10 = get_all_class_obj(padding, 'padding')
    rs11 = get_all_class_obj(pixelshuffle, 'pixielshuffle')
    rs12 = get_all_class_obj(loss, 'loss')

    module_str = """"""
    module_str += "from federatedml.fate_torch.base import FateTorchLayer, FateTorchLoss\nfrom torch import nn\n\n"
    for rs in [rs1, rs2, rs3, rs4, rs5, rs6, rs7, rs8, rs9, rs10, rs11, rs12]:
        module_name = rs[1]
        for i in rs[0]:
            # print(i)
            param = extract_init_param(i)
            class_str = code_assembly(param, i.__name__, module_name)
            module_str += class_str

    module_str = module_str

    open('../nn.py', 'w').write(module_str)
