import random
import torch as t
from torch import nn
import numpy as np
from typing import Literal


def _get_activation(activation):
    if activation == "relu":
        return t.nn.ReLU()
    elif activation == "sigmoid":
        return t.nn.Sigmoid()
    elif activation == "tanh":
        return t.nn.Tanh()
    else:
        raise ValueError(f"Unsupported activation: {activation}")



class PassportBlock(nn.Module):

    def __init__(self, passport_distribute: Literal['gaussian', 'uniform'], passport_mode: Literal['single', 'multi']):

        super().__init__()
        self._bias = None
        self._scale = None
        self._fw_layer = None
        self._passport_distribute = passport_distribute
        self._passport_mode = passport_mode
        assert self._passport_distribute in ['gaussian', 'uniform'], 'passport_distribute must be in ["gaussian", "uniform"]'
        assert self._passport_mode in ['single', 'multi'], 'passport_mode must be in ["single", "multi"]'
        self._encode, self._leaky_relu, self._decode = None, None, None

    def _init_autoencoder(self, in_feat, out_feat):
        self._encode = nn.Linear(in_feat, out_feat, bias=False)
        self._leaky_relu = nn.LeakyReLU(inplace=True)
        self._decode = nn.Linear(out_feat, in_feat, bias=False)

    def _generate_key(self):
        pass

    def set_key(self, skey, bkey):
        self.register_buffer('skey', skey)
        self.register_buffer('bkey', bkey)

    def _compute_para(self, key) -> float:
        pass

    def _get_bias(self):
        return self._compute_para(self.bkey)

    def _get_scale(self):
        return self._compute_para(self.skey)



class LinearPassportBlock(PassportBlock):

    def __init__(self, in_features, out_features, bias=True,
                 passport_distribute: Literal['gaussian', 'uniform'] = 'gaussian',
                 passport_mode: Literal['single', 'multi'] = 'single',
                 loc=-1.0, scale=1.0, low=-1.0, high=1.0, num_passport=1, ae_in=None, ae_out=None):
        super().__init__(
            passport_distribute=passport_distribute,
            passport_mode=passport_mode
        )

        self._num_passport = num_passport
        self._linear = nn.Linear(in_features, out_features, bias=bias)
        if ae_in is None:
            ae_in = out_features
        if ae_out is None:
            ae_out = out_features // 4
        self._init_autoencoder(ae_in, ae_out)
        self.set_key(None, None)
        self._loc = loc
        self._scale = scale
        self._low = low
        self._high = high

        # running var
        self.scale, self.bias = None, None

    def generate_key(self, *shape):

        newshape = list(shape)
        newshape[0] = self.num_passport
        if self.passport_mode == 'single':
            if self.passport_type == 'uniform':
                key = np.random.uniform(self.a, self.b, newshape)
            elif self.passport_type == 'gaussian':
                key = np.random.normal(self.a, self.b, newshape)
            else:
                raise ValueError("Wrong passport type (uniform or gaussian)")

        elif self.passport_mode == 'multi':
            assert self.a != 0
            element_num = newshape[1]  # for every element
            keys = []
            for c in range(element_num):
                if self.a < 0:
                    candidates = range(int(self.a), -1, 1)
                else:
                    candidates = range(1, int(self.a) + 1, 1)
                a = random.sample(candidates, 1)[0]
                while a == 0:
                    a = random.sample(candidates, 1)[0]
                b = self.b
                newshape[1] = 1
                if self.passport_type == 'uniform':
                    key = np.random.uniform(self.a, self.b, newshape)
                elif self.passport_type == 'gaussian':
                    key = np.random.normal(a, b, newshape)
                else:
                    raise ValueError("Wrong passport type (uniform or gaussian)")
                keys.append(key)
            key = np.concatenate(keys, axis=1)
        else:
            raise ValueError("Wrong passport mode, in ['single', 'multi']")
        return key





class ConvPassportBlock(PassportBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True,
                 passport_distribute: Literal['gaussian', 'uniform'] = 'gaussian',
                 passport_mode: Literal['single', 'multi'] = 'single',
                 activation: Literal['relu', 'tanh', 'sigmoid'] = "relu",
                 loc=-1.0, scale=1.0, low=-1.0, high=1.0, num_passport=1, ae_in=None, ae_out=None):

        super().__init__(
            passport_distribute=passport_distribute,
            passport_mode=passport_mode
        )

        self._num_passport = num_passport
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

        if ae_in is None:
            ae_in = out_channels
        if ae_out is None:
            ae_out = out_channels // 4
        self._init_autoencoder(ae_in, ae_out)
        self._bn = nn.BatchNorm2d(out_channels, affine=False)
        self.set_key(None, None)
        self._loc = loc
        self._scale = scale
        self._low = low
        self._high = high
        if activation is not None:
            self._activation = _get_activation(activation)
        else:
            self._activation = None
        # running var
        self.scale, self.bias = None, None

    def generate_key(self, *shape):

        newshape = list(shape)
        newshape[0] = self._num_passport
        if self._passport_mode == 'single':
            if self._passport_distribute == 'uniform':
                key = np.random.uniform(self._low, self._high, newshape)
            elif self._passport_distribute == 'gaussian':
                key = np.random.normal(self._loc, self._scale, newshape)
            else:
                raise ValueError("Wrong passport type (uniform or gaussian)")

        elif self._passport_mode == 'multi':
            assert self._low < self._high
            channel = newshape[1]
            keys = []
            for c in range(channel):
                candidates = range(int(self._low), int(self._high), 1)
                a = random.sample(candidates, 1)[0]
                while a == 0:
                    a = random.sample(candidates, 1)[0]
                b = 1
                newshape[1] = 1
                if self._passport_distribute == 'uniform':
                    key = np.random.uniform(self._low, self._high, newshape)
                elif self._passport_distribute == 'gaussian':
                    key = np.random.normal(a, b, newshape)
                else:
                    raise ValueError("Wrong passport type (uniform or gaussian)")
                keys.append(key)
            key = np.concatenate(keys, axis=1)
        else:
            raise ValueError("Wrong passport mode, in ['single', 'multi']")
        return key

    def _compute_para(self, key):

        b, c, h, w = key.size()
        if c != 1:  # input channel
            randb = random.randint(0, b - 1)
            key = key[randb].unsqueeze(0)
        else:
            key = key
        scalekey = self._conv(key)
        b = scalekey.size(0)
        c = scalekey.size(1)
        scale = scalekey.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        scale = scale.mean(dim=0).view(1, c, 1, 1)
        scale = scale.view(-1, c, 1, 1)
        scale = scale.view(1, c)
        scale = self._decode(self._leaky_relu(self._encode(scale))).view(1, c, 1, 1)
        return scale

    def forward(self, x: t.Tensor):

        if self.skey is None and self.bkey is None:
            skey, bkey = self.generate_key(*x.size()), self.generate_key(*x.size())
            self.set_key(
                t.tensor(skey, dtype=x.dtype, device=x.device),
                t.tensor(bkey, dtype=x.dtype, device=x.device)
            )
        x = self._conv(x)
        x = self._bn(x)
        scale = self._get_scale()
        bias = self._get_bias()
        x = scale * x + bias
        self.scale, self.bias = scale, bias
        return self._activation(x) if self._activation is not None else x



if __name__ == '__main__':

    from torch.nn import init

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, norm_type=None,
                     relu=False):
            super().__init__()

            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
            self.norm_type = norm_type

            if self.norm_type:
                if self.norm_type == 'bn':
                    self.bn = nn.BatchNorm2d(out_channels)
                elif self.norm_type == 'gn':
                    self.bn = nn.GroupNorm(out_channels // 16, out_channels)
                elif self.norm_type == 'in':
                    self.bn = nn.InstanceNorm2d(out_channels)
                else:
                    raise ValueError("Wrong norm_type")
            else:
                self.bn = None

            if relu:
                self.relu = nn.ReLU(inplace=True)
            else:
                self.relu = None

            self.reset_parameters()

        def reset_parameters(self):
            init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        def forward(self, x, scales=None, biases=None):
            x = self.conv(x)
            if self.norm_type is not None:
                x = self.bn(x)
            # print("scales:", scales)
            # print("biases:", biases)
            if scales is not None and biases is not None:
                # print("convent forward")
                x = scales[-1] * x + biases[-1]

            if self.relu is not None:
                x = self.relu(x)
            return x


    class LeNetBottom(nn.Module):
        def __init__(self):
            super(LeNetBottom, self).__init__()
            self.layer0 = nn.Sequential(
                ConvBlock(1, 8, kernel_size=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)
            )

            self.layer1 = nn.Sequential(
                ConvPassportBlock(8, 16, 5, num_passport=64),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            x = self.layer0(x)
            x = self.layer1(x)
            return x

    import torchvision
    # get mnist dataset
    train_data = torchvision.datasets.MNIST(root='/home/cwj/mnist',
                                            train=True, download=True, transform=torchvision.transforms.ToTensor())
    # put in train loader batch size = 8
    train_loader = t.utils.data.DataLoader(dataset=train_data, batch_size=8, shuffle=True)
    for i, (x, y) in enumerate(train_loader):
        print(x.shape)
        break

    model = LeNetBottom()
    out_ = model(x)