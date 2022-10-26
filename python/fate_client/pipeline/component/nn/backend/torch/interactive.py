import torch as t
from pipeline.component.nn.backend.torch.nn import Linear, LazyLinear, ReLU, Sigmoid, Tanh
from pipeline.component.nn.backend.torch.base import FateTorchLayer


class InteractiveLayer(t.nn.Module, FateTorchLayer):

    r"""A :class: InteractiveLayer.

           An interface for InteractiveLayer. In interactive layer, the forward method is:
           out = activation( Linear(guest_input) + Linear(host_0_input) + Linear(host_1_input) ..)

           Args:
                out_dim: int, the output dimension of InteractiveLayer
                host_num: int, specify the number of host party, default is 1, need to modify this parameter
                               when running multi-party modeling
                guest_dim: int or None, the input dimension of guest features, if None, will use LazyLinear layer
                           that automatically infers the input dimension
                host_dim: int, or None:
                           int: the input dimension of all host features
                           None: automatically infer the input dimension of all host features
                activation: str, support relu, tanh, sigmoid
                guest_bias: bias for guest linear layer
                host_bias: bias for host linear layers
           """

    def __init__(self, out_dim, guest_dim=None, host_num=1, host_dim=None, activation='relu', guest_bias=True,
                 host_bias=True):

        t.nn.Module.__init__(self)
        FateTorchLayer.__init__(self)
        self.activation = None
        if activation is not None:
            if activation.lower() == 'relu':
                self.activation = ReLU()
            elif activation.lower() == 'tanh':
                self.activation = Tanh()
            elif activation.lower() == 'sigmoid':
                self.activation = Sigmoid()
            else:
                raise ValueError('activation not support {}, avail: relu, tanh, sigmoid'.format(activation))

        assert isinstance(out_dim, int), 'out_dim must be an int >= 0'

        self.param_dict['out_dim'] = out_dim
        self.param_dict['activation'] = activation

        assert isinstance(host_num, int) and host_num >= 1, 'host number is an int >= 1'
        self.param_dict['host_num'] = host_num

        if guest_dim is not None:
            assert isinstance(guest_dim, int)
        if host_dim is not None:
            assert isinstance(host_dim, int)

        self.param_dict['guest_dim'] = guest_dim
        self.param_dict['host_dim'] = host_dim
        self.param_dict['guest_bias'] = guest_bias
        self.param_dict['host_bias'] = host_bias

        if guest_dim is None:
            self.guest_model = LazyLinear(out_dim, guest_bias)
        else:
            self.guest_model = Linear(guest_dim, out_dim, guest_bias)

        self.out_dim = out_dim
        self.host_dim = host_dim
        self.host_bias = host_bias
        self.host_model = None

        self.host_model = t.nn.ModuleList()
        for i in range(host_num):
            self.host_model.append(self.make_host_model())

    def make_host_model(self):
        if self.host_dim is None:
            return LazyLinear(self.out_dim, self.host_bias)
        else:
            return Linear(self.host_dim, self.out_dim, self.host_bias)

    def forward(self, x_guest, x_host):

        g_out = self.guest_model(x_guest)
        h_out = None
        if isinstance(x_host, list):
            for m, data in zip(self.host_model, x_host):
                out_ = m(data)
                if h_out is None:
                    h_out = out_
                else:
                    h_out += out_
        else:
            h_out = self.host_model[0](x_host)

        return self.activation(g_out + h_out)

