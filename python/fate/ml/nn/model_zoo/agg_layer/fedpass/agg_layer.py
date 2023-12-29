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
from fate.ml.nn.model_zoo.agg_layer.agg_layer import AggLayerHost, AggLayerGuest
from fate.ml.nn.model_zoo.agg_layer.fedpass._passport_block import ConvPassportBlock, LinearPassportBlock


def get_model(
    layer_type: Literal["conv", "linear"],
    in_channels_or_features: int,
    out_channels_or_features: int,
    kernel_size: Union[int, Tuple[int, int]] = 3,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    bias: bool = True,
    hidden_features: int = 128,
    activation: Literal["relu", "tanh", "sigmoid"] = "relu",
    passport_distribute: Literal["gaussian", "uniform"] = "gaussian",
    passport_mode: Literal["single", "multi"] = "single",
    loc=-1.0,
    scale=1.0,
    low=-1.0,
    high=1.0,
    num_passport=1,
    ae_in=None,
    ae_out=None,
    **kwargs,
):
    if layer_type == "conv":
        model = ConvPassportBlock(
            in_channels=in_channels_or_features,
            out_channels=out_channels_or_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            passport_distribute=passport_distribute,
            passport_mode=passport_mode,
            loc=loc,
            scale=scale,
            low=low,
            high=high,
            num_passport=num_passport,
            ae_in=ae_in,
            ae_out=ae_out,
            activation=activation,
        )
    elif layer_type == "linear":
        model = LinearPassportBlock(
            in_features=in_channels_or_features,
            out_features=out_channels_or_features,
            bias=bias,
            passport_distribute=passport_distribute,
            passport_mode=passport_mode,
            loc=loc,
            scale=scale,
            low=low,
            high=high,
            num_passport=num_passport,
            ae_in=ae_in,
            ae_out=ae_out,
            hidden_feature=hidden_features,
        )
    else:
        raise ValueError(f"Unsupported layer type: {layer_type}, available: ['conv', 'linear']")

    return model


class FedPassAggLayerGuest(AggLayerGuest):
    def __init__(
        self,
        layer_type: Literal["conv", "linear"],
        in_channels_or_features: int,
        out_channels_or_features: int,
        kernel_size: Union[int, tuple] = 3,
        stride: Union[int, tuple] = 1,
        padding: int = 0,
        bias: bool = True,
        hidden_features: int = 128,
        activation: Literal["relu", "tanh", "sigmoid"] = "relu",
        passport_distribute: Literal["gaussian", "uniform"] = "gaussian",
        passport_mode: Literal["single", "multi"] = "single",
        loc=-1.0,
        scale=1.0,
        low=-1.0,
        high=1.0,
        num_passport=1,
        ae_in=None,
        ae_out=None,
        merge_type: Literal["sum", "concat"] = "sum",
        concat_dim=1,
        **kwargs,
    ):
        super().__init__(merge_type, concat_dim)

        model = get_model(
            layer_type=layer_type,
            in_channels_or_features=in_channels_or_features,
            out_channels_or_features=out_channels_or_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            hidden_features=hidden_features,
            activation=activation,
            passport_distribute=passport_distribute,
            passport_mode=passport_mode,
            loc=loc,
            scale=scale,
            low=low,
            high=high,
            num_passport=num_passport,
            ae_in=ae_in,
            ae_out=ae_out,
        )

        self._model = model


class FedPassAggLayerHost(AggLayerHost):
    def __init__(
        self,
        layer_type: Literal["conv", "linear"],
        in_channels_or_features: int,
        out_channels_or_features: int,
        kernel_size: Union[int, tuple] = 3,
        stride: Union[int, tuple] = 1,
        padding: int = 0,
        bias: bool = True,
        hidden_features: int = 128,
        activation: Literal["relu", "tanh", "sigmoid"] = "relu",
        passport_distribute: Literal["gaussian", "uniform"] = "gaussian",
        passport_mode: Literal["single", "multi"] = "single",
        loc=-1.0,
        scale=1.0,
        low=-1.0,
        high=1.0,
        num_passport=1,
        ae_in=None,
        ae_out=None,
        **kwargs,
    ):
        super(FedPassAggLayerHost, self).__init__()

        model = get_model(
            layer_type=layer_type,
            in_channels_or_features=in_channels_or_features,
            out_channels_or_features=out_channels_or_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            hidden_features=hidden_features,
            activation=activation,
            passport_distribute=passport_distribute,
            passport_mode=passport_mode,
            loc=loc,
            scale=scale,
            low=low,
            high=high,
            num_passport=num_passport,
            ae_in=ae_in,
            ae_out=ae_out,
        )

        self._model = model
