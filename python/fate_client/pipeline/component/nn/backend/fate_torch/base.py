import json
try:
    import torch as t
    from torch.nn import Sequential as tSequential
except ImportError:
    t = None
    tSequential = object


class FateTorchLayer(object):

    def __init__(self):
        t.nn.Module.__init__(self)
        self.param_dict = dict()
        self.initializer = {'weight': None, 'bias': None}
        self.optimizer = None

    def to_dict(self):
        import copy
        ret_dict = copy.deepcopy(self.param_dict)
        ret_dict['layer'] = type(self).__name__
        ret_dict['initializer'] = {}
        if self.initializer['weight']:
            ret_dict['initializer']['weight'] = self.initializer['weight']
        if self.initializer['bias']:
            ret_dict['initializer']['bias'] = self.initializer['bias']
        return ret_dict

    def add_optimizer(self, opt):
        self.optimizer = opt


class FateTorchLoss(object):

    def __init__(self):
        self.param_dict = {}

    def to_dict(self):
        import copy
        ret_dict = copy.deepcopy(self.param_dict)
        ret_dict['loss_fn'] = type(self).__name__
        return ret_dict


class FateTorchOptimizer(object):

    def __init__(self):
        self.param_dict = dict()
        self.torch_class = None

    def to_dict(self):
        import copy
        ret_dict = copy.deepcopy(self.param_dict)
        ret_dict['optimizer'] = type(self).__name__
        ret_dict['config_type'] = 'pytorch'
        return ret_dict

    def register_optimizer(self, fate_torch_component):
        if fate_torch_component is None:
            return
        if not isinstance(fate_torch_component, FateTorchLayer) and not isinstance(fate_torch_component, Sequential):
            raise ValueError('Please pass FateTorchLayer or Sequential to register optimizer')
        fate_torch_component.add_optimizer(self)

    def to_torch_instance(self, parameters):
        return self.torch_class(parameters, **self.param_dict)


class Sequential(tSequential):

    def to_dict(self):
        """
        get the structure of current sequential
        """
        rs = {}
        idx = 0
        for k in self._modules:
            ordered_name = str(idx) + '-' + k
            rs[ordered_name] = self._modules[k].to_dict()
            idx += 1
        return rs

    def to_json(self):
        return json.dumps(self.to_dict(), indent=4)

    def add_optimizer(self, opt):
        setattr(self, 'optimizer', opt)

    def add(self, layer):

        if isinstance(layer, Sequential):
            self._modules = layer._modules
            # copy optimizer
            if hasattr(layer, 'optimizer'):
                setattr(self, 'optimizer', layer.optimizer)
        elif isinstance(layer, FateTorchLayer):
            self.add_module(str(len(self)), layer)
            # update optimizer if dont have
            if not hasattr(self, 'optimizer') and hasattr(layer, 'optimizer'):
                setattr(self, 'optimizer', layer.optimizer)
        else:
            raise ValueError('unknown input layer type {}, this type is not supported'.format(type(layer)))

    @staticmethod
    def get_loss_config(loss: FateTorchLoss):
        return loss.to_dict()

    def get_optimizer_config(self, optimizer=None):
        if hasattr(self, 'optimizer'):
            return self.optimizer.to_dict()
        else:
            return optimizer.to_dict()

    def get_network_config(self):
        return self.to_dict()
