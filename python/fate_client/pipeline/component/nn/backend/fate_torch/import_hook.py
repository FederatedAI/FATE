try:
    from pipeline.component.nn.backend.fate_torch import nn as nn_
    from pipeline.component.nn.backend.fate_torch import init as init_
    from pipeline.component.nn.backend.fate_torch import optim as optim_
except ImportError:
    pass


def monkey_patch(torch_nn, fate_torch_module):
    for name in fate_torch_module.__dict__.keys():
        if '__' in name:  # skip no related variables
            continue
        if name in torch_nn.__dict__.keys():
            torch_nn.__dict__[name] = fate_torch_module.__dict__[name]


def fate_torch_hook(torch_module_var):
    """
    This is a monkey patch function that modify torch modules to use fate_torch layers and Components
    :param torch_module_var:
    :return:
    """
    if torch_module_var.__name__ == 'torch':

        monkey_patch(torch_module_var.nn, nn_)
        monkey_patch(torch_module_var.optim, optim_)
        monkey_patch(torch_module_var.nn.init, init_)

    elif torch_module_var.__name__ == 'torch.nn':
        monkey_patch(torch_module_var, nn_)

    elif torch_module_var.__name__ == 'torch.optim':
        monkey_patch(torch_module_var, optim_)

    elif torch_module_var.__name__ == 'torch.nn.init':
        monkey_patch(torch_module_var, init_)

    else:
        raise ValueError('this module: {} does not support fate torch hook'.format(torch_module_var))

    return torch_module_var
