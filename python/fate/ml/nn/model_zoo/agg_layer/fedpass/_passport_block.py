#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

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
    def __init__(self, passport_distribute: Literal["gaussian", "uniform"], passport_mode: Literal["single", "multi"]):
        super().__init__()
        self._fw_layer = None
        self._passport_distribute = passport_distribute
        self._passport_mode = passport_mode
        self._out_scale, self._out_bias = None, None
        assert self._passport_distribute in [
            "gaussian",
            "uniform",
        ], 'passport_distribute must be in ["gaussian", "uniform"]'
        assert self._passport_mode in ["single", "multi"], 'passport_mode must be in ["single", "multi"]'
        self._encode, self._leaky_relu, self._decode = None, None, None

    def _init_autoencoder(self, in_feat, out_feat):
        self._encode = nn.Linear(in_feat, out_feat, bias=False)
        self._leaky_relu = nn.LeakyReLU(inplace=True)
        self._decode = nn.Linear(out_feat, in_feat, bias=False)

    def _generate_key(self):
        pass

    def set_key(self, skey, bkey):
        self.register_buffer("skey", skey)
        self.register_buffer("bkey", bkey)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        skey = "_agg_layer._model.skey"
        bkey = "_agg_layer._model.bkey"

        if skey in state_dict:
            self.register_buffer("skey", t.randn(*state_dict[skey].size()))
        if bkey in state_dict:
            self.register_buffer("bkey", t.randn(*state_dict[bkey].size()))

        if "_out_scale" in state_dict:
            self.scale = nn.Parameter(t.randn(*state_dict["_out_scale"].size()))

        if "_out_bias" in state_dict:
            self.bias = nn.Parameter(t.randn(*state_dict["_out_bias"].size()))

        super(PassportBlock, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def _compute_para(self, key) -> float:
        pass

    def _get_bias(self):
        return self._compute_para(self.bkey)

    def _get_scale(self):
        return self._compute_para(self.skey)


class LinearPassportBlock(PassportBlock):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        hidden_feature: int = 128,
        passport_distribute: Literal["gaussian", "uniform"] = "gaussian",
        passport_mode: Literal["single", "multi"] = "single",
        loc=-1.0,
        scale=1.0,
        low=-10.0,
        high=1.0,
        num_passport=1,
        ae_in=None,
        ae_out=None,
    ):
        super().__init__(passport_distribute=passport_distribute, passport_mode=passport_mode)

        self._num_passport = num_passport
        self._linear = nn.Linear(in_features, hidden_feature, bias=bias)
        self._linear2 = nn.Linear(hidden_feature, out_features, bias=bias)
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
        newshape[0] = self._num_passport
        if self._passport_mode == "single":
            if self._passport_distribute == "uniform":
                key = np.random.uniform(self._low, self._high, newshape)
            elif self._passport_distribute == "gaussian":
                key = np.random.normal(self._loc, self._scale, newshape)
            else:
                raise ValueError("Wrong passport type (uniform or gaussian)")

            return key

        elif self._passport_mode == "multi":
            assert self._low != 0
            element = newshape[1]
            keys = []
            for c in range(element):
                if self._low < 0:
                    candidates = range(int(self._low), -1, 1)
                else:
                    candidates = range(1, int(self._low) + 1, 1)
                a = random.sample(candidates, 1)[0]
                while a == 0:
                    a = random.sample(candidates, 1)[0]
                b = self._high
                newshape[1] = 1
                if self._passport_distribute == "uniform":
                    key = np.random.uniform(self._low, self._high, newshape)
                elif self._passport_distribute == "gaussian":
                    key = np.random.normal(a, b, newshape)
                else:
                    raise ValueError("Wrong passport type (uniform or gaussian)")
                keys.append(key)

            key = np.concatenate(keys, axis=1)
            return key
        else:
            raise ValueError("Wrong passport mode, in ['single', 'multi']")

    def _compute_para(self, key) -> float:
        scalekey = self._linear(key)
        scalekey = self._linear2(scalekey)
        b = scalekey.size(0)
        c = scalekey.size(1)
        scale = scalekey.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        scale = scale.mean(dim=0).view(-1, c, 1, 1)
        scale = scale.view(-1, c)
        scale = scale.view(1, c)
        scale = self._decode(self._leaky_relu(self._encode(scale))).view(1, c)
        return scale

    def forward(self, x: t.Tensor):
        if self.skey is None and self.bkey is None:
            skey, bkey = self.generate_key(*x.size()), self.generate_key(*x.size())
            self.set_key(
                t.tensor(skey, dtype=x.dtype, device=x.device), t.tensor(bkey, dtype=x.dtype, device=x.device)
            )
        x = self._linear(x)
        x = self._linear2(x)
        scale = self._get_scale()
        bias = self._get_bias()
        self._out_scale, self._out_bias = scale, bias
        x = t.nn.Tanh()(scale) * x + t.nn.Tanh()(bias)
        return x


class ConvPassportBlock(PassportBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        passport_distribute: Literal["gaussian", "uniform"] = "gaussian",
        passport_mode: Literal["single", "multi"] = "single",
        activation: Literal["relu", "tanh", "sigmoid"] = "relu",
        loc=-1.0,
        scale=1.0,
        low=-1.0,
        high=1.0,
        num_passport=1,
        ae_in=None,
        ae_out=None,
    ):
        super().__init__(passport_distribute=passport_distribute, passport_mode=passport_mode)

        self._num_passport = num_passport
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
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

    def generate_key(self, *shape):
        newshape = list(shape)
        newshape[0] = self._num_passport
        if self._passport_mode == "single":
            if self._passport_distribute == "uniform":
                key = np.random.uniform(self._low, self._high, newshape)
            elif self._passport_distribute == "gaussian":
                key = np.random.normal(self._loc, self._scale, newshape)
            else:
                raise ValueError("Wrong passport type (uniform or gaussian)")

        elif self._passport_mode == "multi":
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
                if self._passport_distribute == "uniform":
                    key = np.random.uniform(self._low, self._high, newshape)
                elif self._passport_distribute == "gaussian":
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
                t.tensor(skey, dtype=x.dtype, device=x.device), t.tensor(bkey, dtype=x.dtype, device=x.device)
            )
        x = self._conv(x)
        x = self._bn(x)
        scale = self._get_scale()
        bias = self._get_bias()
        x = scale * x + bias
        self._out_scale, self._out_bias = scale, bias
        return self._activation(x) if self._activation is not None else x
