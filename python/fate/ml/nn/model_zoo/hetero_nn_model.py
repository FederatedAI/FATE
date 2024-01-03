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

import torch
import torch as t
from fate.arch import Context
from fate.ml.nn.model_zoo.agg_layer.agg_layer import AggLayerGuest, AggLayerHost
from fate.ml.nn.model_zoo.agg_layer.fedpass.agg_layer import FedPassAggLayerGuest, FedPassAggLayerHost, get_model
from fate.ml.nn.model_zoo.agg_layer.sshe.agg_layer import SSHEAggLayerHost, SSHEAggLayerGuest
from typing import Any, Dict, List, Union, Callable, Literal
from dataclasses import dataclass, fields
from enum import Enum
import logging
from torch import device

logger = logging.getLogger(__name__)


"""
Agg Layer Arguments
"""


@dataclass
class Args(object):
    def to_dict(self):
        d = dict((field.name, getattr(self, field.name)) for field in fields(self) if field.init)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class StdAggLayerArgument(Args):
    merge_type: Literal["sum", "concat"] = "sum"
    concat_dim = 1

    def to_dict(self):
        d = super().to_dict()
        d["agg_type"] = "std"
        return d


@dataclass
class FedPassArgument(StdAggLayerArgument):
    layer_type: Literal["conv", "linear"] = "conv"
    in_channels_or_features: int = 8
    out_channels_or_features: int = 8
    kernel_size: Union[int, tuple] = 3
    stride: Union[int, tuple] = 1
    padding: int = 0
    bias: bool = True
    hidden_features: int = 128
    activation: Literal["relu", "tanh", "sigmoid"] = "relu"
    passport_distribute: Literal["gaussian", "uniform"] = "gaussian"
    passport_mode: Literal["single", "multi"] = "single"
    loc: int = -1.0
    scale: int = 1.0
    low: int = -1.0
    high: int = 1.0
    num_passport: int = 1
    ae_in: int = None
    ae_out: int = None

    def to_dict(self):
        d = super().to_dict()
        d["agg_type"] = "fed_pass"
        return d


@dataclass
class SSHEArgument(Args):
    guest_in_features: int = 8
    host_in_features: int = 8
    out_features: int = 8
    layer_lr: float = 0.01
    precision_bits: int = None

    def to_dict(self):
        d = super().to_dict()
        d["agg_type"] = "hess"
        return d


def parse_agglayer_conf(agglayer_arg_conf):
    import copy

    if "agg_type" not in agglayer_arg_conf:
        raise ValueError("can not load agg layer conf, keyword agg_type not found")
    agglayer_arg_conf = copy.deepcopy(agglayer_arg_conf)
    agg_type = agglayer_arg_conf["agg_type"]
    agglayer_arg_conf.pop("agg_type")
    if agg_type == "fed_pass":
        agglayer_arg = FedPassArgument(**agglayer_arg_conf)
    elif agg_type == "std":
        agglayer_arg = StdAggLayerArgument(**agglayer_arg_conf)
    elif agg_type == "hess":
        agglayer_arg = SSHEArgument(**agglayer_arg_conf)
    else:
        raise ValueError(f"agg type {agg_type} not supported")

    return agglayer_arg


"""
Top & Bottom Model Strategy
"""


@dataclass
class TopModelStrategyArguments(Args):
    protect_strategy: Literal["fedpass"] = None
    fed_pass_arg: Union[FedPassArgument, dict] = None
    add_output_layer: Literal[None, "sigmoid", "softmax"] = None

    def __post_init__(self):
        if self.protect_strategy == "fedpass":
            if isinstance(self.fed_pass_arg, dict):
                self.fed_pass_arg = FedPassArgument(**self.fed_pass_arg)
            if not isinstance(self.fed_pass_arg, FedPassArgument):
                raise TypeError("fed_pass_arg must be an instance of FedPassArgument for protect_strategy 'fedpass'")

        assert self.add_output_layer in [
            None,
            "sigmoid",
            "softmax",
        ], "add_output_layer must be None, 'sigmoid' or 'softmax'"

    def to_dict(self):
        d = super().to_dict()
        if "fed_pass_arg" in d:
            d["fed_pass_arg"] = d["fed_pass_arg"].to_dict()
            d["fed_pass_arg"].pop("agg_type")
        return d


def backward_loss(z, backward_error):
    return t.sum(z * backward_error)


class HeteroNNModelBase(t.nn.Module):
    def __init__(self):
        super().__init__()
        self._bottom_model = None
        self._top_model = None
        self._agg_layer = None
        self._ctx = None
        self.device = None

    def _auto_setup(self):
        self._agg_layer = AggLayerGuest()
        self._agg_layer.set_context(self._ctx)

    def get_device(self, module):
        return next(module.parameters()).device


class HeteroNNModelGuest(HeteroNNModelBase):
    def __init__(
        self,
        top_model: t.nn.Module,
        bottom_model: t.nn.Module = None,
        agglayer_arg: Union[StdAggLayerArgument, FedPassArgument, SSHEArgument] = None,
        top_arg: TopModelStrategyArguments = None,
        ctx: Context = None,
    ):
        super(HeteroNNModelGuest, self).__init__()
        # cached variables
        if top_model is None:
            raise RuntimeError("guest needs a top model to compute loss, but no top model provided")
        assert isinstance(top_model, t.nn.Module), "top model should be a torch nn.Module"
        self._top_model = top_model
        self._agg_layer = None
        if bottom_model is not None:
            assert isinstance(bottom_model, t.nn.Module), "bottom model should be a torch nn.Module"
            self._bottom_model = bottom_model

        self._bottom_fw = None  # for backward usage
        self._agg_fw_rg = None  # for backward usage
        # ctx
        self._ctx = None
        # internal mode
        self._guest_direct_backward = True
        # set top strategy
        self._top_strategy = None
        # top additional model
        self._top_add_model = None
        self.setup(ctx=ctx, agglayer_arg=agglayer_arg, top_arg=top_arg, bottom_arg=None)

    def __repr__(self):
        return (
            f"HeteroNNGuest(top_model={self._top_model}\n"
            f"agg_layer={self._agg_layer}\n"
            f"bottom_model={self._bottom_model})"
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _clear_state(self):
        self._bottom_fw = None
        self._agg_fw_rg = None

    def need_mpc_init(self):
        return isinstance(self._agg_layer, SSHEAggLayerGuest)

    def setup(
        self,
        ctx: Context = None,
        agglayer_arg: Union[StdAggLayerArgument, FedPassArgument, SSHEArgument] = None,
        top_arg: TopModelStrategyArguments = None,
        bottom_arg=None,
    ):
        self._ctx = ctx

        if self._agg_layer is None:
            if agglayer_arg is None:
                self._agg_layer = AggLayerGuest()
            elif type(agglayer_arg) == StdAggLayerArgument:
                self._agg_layer = AggLayerGuest(**agglayer_arg.to_dict())
            elif type(agglayer_arg) == FedPassArgument:
                self._agg_layer = FedPassAggLayerGuest(**agglayer_arg.to_dict())
            elif type(agglayer_arg) == SSHEArgument:
                self._agg_layer = SSHEAggLayerGuest(**agglayer_arg.to_dict())
                if self._bottom_model is None:
                    raise RuntimeError("A bottom model is needed when running a SSHE model")

        if self._top_add_model is None:
            if top_arg:
                logger.info("detect top model strategy")
                if top_arg.protect_strategy == "fedpass":
                    fedpass_arg = top_arg.fed_pass_arg
                    top_fedpass_model = get_model(**fedpass_arg.to_dict())
                    self._top_add_model = top_fedpass_model
                    self._top_model = t.nn.Sequential(self._top_model, top_fedpass_model)
                    if top_arg.add_output_layer == "sigmoid":
                        self._top_model.add_module("sigmoid", t.nn.Sigmoid())
                    elif top_arg.add_output_layer == "softmax":
                        self._top_model.add_module("softmax", t.nn.Softmax(dim=1))

        self._agg_layer.set_context(ctx)

    def forward(self, x=None):
        if self._agg_layer is None:
            self._auto_setup()

        if self.device is None:
            self.device = self.get_device(self._top_model)
            self._agg_layer.set_device(self.device)
            if isinstance(self._agg_layer, SSHEAggLayerHost):
                if self.device.type != "cpu":
                    raise ValueError("SSHEAggLayerGuest is not supported on GPU")

        if self._bottom_model is None:
            b_out = None
        else:
            b_out = self._bottom_model(x)
            # bottom layer
            if not self._guest_direct_backward:
                self._bottom_fw = b_out

        # hetero layer
        if not self._guest_direct_backward:
            agg_out = self._agg_layer.forward(b_out)
            self._agg_fw_rg = agg_out.requires_grad_(True)
            # top layer
            top_out = self._top_model(self._agg_fw_rg)
        else:
            top_out = self._top_model(self._agg_layer(b_out))

        return top_out

    def backward(self, loss):
        if self._guest_direct_backward:
            # send error to hosts & guest side direct backward
            if isinstance(self._agg_layer, SSHEAggLayerGuest):
                loss.backward()  # sshe has independent optimizer
                self._agg_layer.step()
            else:
                self._agg_layer.backward(loss)
                loss.backward()
        else:
            # backward are split into parts
            loss.backward()  # update top
            agg_error = self._agg_fw_rg.grad
            self._agg_fw_rg = None
            bottom_error = self._agg_layer.backward(agg_error)  # compute bottom error & update hetero
            if bottom_error is not None:
                bottom_loss = backward_loss(self._bottom_fw, bottom_error)
                bottom_loss.backward()
            self._bottom_fw = False

    def predict(self, x=None):
        with torch.no_grad():
            if self._bottom_model is None:
                b_out = None
            else:
                b_out = self._bottom_model(x)
            agg_out = self._agg_layer.predict(b_out)
            top_out = self._top_model(agg_out)

        return top_out

    def _auto_setup(self):
        self._agg_layer = AggLayerGuest()
        self._agg_layer.set_context(self._ctx)


class HeteroNNModelHost(HeteroNNModelBase):
    def __init__(
        self,
        bottom_model: t.nn.Module,
        agglayer_arg: Union[StdAggLayerArgument, FedPassArgument, SSHEArgument] = None,
        ctx: Context = None,
    ):
        super().__init__()

        assert isinstance(bottom_model, t.nn.Module), "bottom model should be a torch nn.Module"
        self._bottom_model = bottom_model
        # cached variables
        self._bottom_fw = None  # for backward usage
        # ctx
        self._ctx = None
        self._agg_layer = None
        self._fake_loss = None
        self.setup(ctx=ctx, agglayer_arg=agglayer_arg)

    def __repr__(self):
        return f"HeteroNNHost(bottom_model={self._bottom_model}, agg_layer={self._agg_layer})"

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def _clear_state(self):
        self._bottom_fw = None

    def need_mpc_init(self):
        return isinstance(self._agg_layer, SSHEAggLayerHost)

    def setup(
        self,
        ctx: Context = None,
        agglayer_arg: Union[StdAggLayerArgument, FedPassArgument, SSHEArgument] = None,
        bottom_arg=None,
    ):
        self._ctx = ctx

        if self._agg_layer is None:
            if agglayer_arg is None:
                self._agg_layer = AggLayerHost()
            elif type(agglayer_arg) == StdAggLayerArgument:
                self._agg_layer = AggLayerHost()  # no parameters are needed
            elif type(agglayer_arg) == FedPassArgument:
                self._agg_layer = FedPassAggLayerHost(**agglayer_arg.to_dict())
            elif isinstance(agglayer_arg, SSHEArgument):
                self._agg_layer = SSHEAggLayerHost(**agglayer_arg.to_dict())

        self._agg_layer.set_context(ctx)

    def forward(self, x):
        if self._agg_layer is None:
            self._auto_setup()

        if self.device is None:
            self.device = self.get_device(self._bottom_model)
            self._agg_layer.set_device(self.device)
            if isinstance(self._agg_layer, SSHEAggLayerHost):
                if self.device.type != "cpu":
                    raise ValueError("SSHEAggLayerGuest is not supported on GPU")

        b_out = self._bottom_model(x)
        # bottom layer
        self._bottom_fw = b_out
        # hetero layer
        if isinstance(self._agg_layer, SSHEAggLayerHost):
            self._fake_loss = self._agg_layer.forward(b_out)
        else:
            self._agg_layer.forward(b_out)

    def backward(self):
        if isinstance(self._agg_layer, SSHEAggLayerHost):
            self._fake_loss.backward()
            self._fake_loss = None
            self._agg_layer.step()  # sshe has independent optimizer
            self._clear_state()
        else:
            error = self._agg_layer.backward()
            error = error.to(self.device)
            loss = backward_loss(self._bottom_fw, error)
            loss.backward()
            self._clear_state()

    def predict(self, x):
        with torch.no_grad():
            b_out = self._bottom_model(x)
            self._agg_layer.predict(b_out)

    def _auto_setup(self):
        self._agg_layer = AggLayerHost()
        self._agg_layer.set_context(self._ctx)
