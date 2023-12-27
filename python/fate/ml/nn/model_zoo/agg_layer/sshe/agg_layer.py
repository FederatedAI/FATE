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

from typing import Union, Literal, Tuple
import torch
from fate.arch import Context
from fate.ml.nn.model_zoo.agg_layer.agg_layer import _AggLayerBase
from fate.arch.protocol.mpc.nn.sshe.nn_layer import SSHENeuralNetworkAggregatorLayer, SSHENeuralNetworkOptimizerSGD


def setup_model(self, ctx: Context):
    generator = torch.Generator()
    self._model = SSHENeuralNetworkAggregatorLayer(
        ctx,
        in_features_a=self._host_in_features,
        in_features_b=self._guest_in_features,
        out_features=self._out_features,
        rank_a=ctx.hosts[0].rank,
        rank_b=ctx.guest.rank,
        wa_init_fn=lambda shape: torch.rand(shape, generator=generator),
        wb_init_fn=lambda shape: torch.rand(shape, generator=generator),
        precision_bits=self._precision_bits,
    )
    self._optimizer = SSHENeuralNetworkOptimizerSGD(ctx, self._model.parameters(), lr=self._layer_lr)


class SSHEAggLayerGuest(_AggLayerBase):
    def __init__(
        self, guest_in_features=4, host_in_features=4, out_features=4, layer_lr=0.01, precision_bits=None, **kwargs
    ):
        super().__init__()
        self._model = None
        self._layer_lr = layer_lr
        self._guest_in_features = guest_in_features
        self._host_in_features = host_in_features
        self._out_features = out_features
        self._optimizer = None
        self._precision_bits = precision_bits

    def set_context(self, ctx: Context):
        if isinstance(ctx, Context):
            super().set_context(ctx)
            setup_model(self, ctx)
            if hasattr(self, "wa") and hasattr(self, "wb"):
                self._set_wa_wb()
            else:
                self.register_buffer("wa", self._model.aggregator.wa.share)
                self.register_buffer("wb", self._model.aggregator.wb.share)

    def _set_wa_wb(self):
        if self._model is not None:
            if hasattr(self, "wa") and hasattr(self, "wb"):
                self._model.aggregator.wa.share = self.wa
                self._model.aggregator.wb.share = self.wb

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if "_agg_layer.wa" in state_dict:
            self.register_buffer("wa", torch.randn(*state_dict["_agg_layer.wa"].size()).type(torch.LongTensor))
        if "_agg_layer.wb" in state_dict:
            self.register_buffer("wb", torch.randn(*state_dict["_agg_layer.wb"].size()).type(torch.LongTensor))

        super(SSHEAggLayerGuest, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        self._set_wa_wb()

    def forward(self, x):
        if self._model is None:
            raise RuntimeError("Model is not initialized! Call set_context to initialize ctx and model")
        fw_rs = self._model(x)
        return fw_rs

    def backward(self, error=None):
        pass

    def step(self):
        self._optimizer.step()


class SSHEAggLayerHost(_AggLayerBase):
    def __init__(
        self, guest_in_features=4, host_in_features=4, out_features=4, layer_lr=0.01, precision_bits=None, **kwargs
    ):
        super().__init__()
        self._model = None
        self._layer_lr = layer_lr
        self._guest_in_features = guest_in_features
        self._host_in_features = host_in_features
        self._out_features = out_features
        self._optimizer = None
        self._precision_bits = precision_bits

    def set_context(self, ctx: Context):
        if isinstance(ctx, Context):
            super().set_context(ctx)
            setup_model(self, ctx)
            if hasattr(self, "wa") and hasattr(self, "wb"):
                self._set_wa_wb()
            else:
                self.register_buffer("wa", self._model.aggregator.wa.share)
                self.register_buffer("wb", self._model.aggregator.wb.share)

    def _set_wa_wb(self):
        if self._model is not None:
            if hasattr(self, "wa") and hasattr(self, "wb"):
                self._model.aggregator.wa.share = self.wa
                self._model.aggregator.wb.share = self.wb

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if "_agg_layer.wa" in state_dict:
            self.register_buffer("wa", torch.randn(*state_dict["_agg_layer.wa"].size()).type(torch.LongTensor))
        if "_agg_layer.wb" in state_dict:
            self.register_buffer("wb", torch.randn(*state_dict["_agg_layer.wb"].size()).type(torch.LongTensor))

        super(SSHEAggLayerHost, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

        self._set_wa_wb()

    def forward(self, x):
        if self._model is None:
            raise RuntimeError("Model is not initialized! Call set_context to initialize ctx and model")
        fake_loss = self._model(x)
        return fake_loss

    def backward(self, error=None):
        pass

    def step(self):
        self._optimizer.step()
