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
#
import inspect

from federatedml.nn.backend.pytorch.custom import layer as custom_layer
from torch import nn
import torch

from federatedml.util import LOGGER

_support_layers = {
    "Linear": nn.Linear,
    "Conv2d": nn.Conv2d,
    "MaxPool2d": nn.MaxPool2d,
    "Flatten": nn.Flatten,
    "ReLU": nn.ReLU,
}


def _supported_layers():
    layers = list(_support_layers.keys())
    for module in [custom_layer]:
        for name, _ in inspect.getmembers(
            module,
            lambda cls: inspect.isclass(cls) and issubclass(cls, nn.Module),
        ):
            if name.startswith("_"):
                continue
            layers.append(name)
    return layers


def get_layer_cls(layer_name):
    if layer_name in _support_layers:
        return _support_layers[layer_name]

    modules = [torch.nn, custom_layer]
    for i, module in enumerate(modules):
        if hasattr(module, layer_name) and issubclass(
            getattr(module, layer_name), nn.Module
        ):
            return getattr(module, layer_name)
        LOGGER.debug(
            f"layer cls {layer_name} not found in {module}, searching {modules[i + 1:]}"
        )
    raise PyTorchLayerNotFoundException(
        f"layer named `{layer_name}` not found, use one of `{_supported_layers()}`"
    )


def get_layer_fn(layer_name, layer_kwargs):
    optimizer_cls = get_layer_cls(layer_name)
    try:
        return optimizer_cls(**layer_kwargs)
    except Exception as e:
        signature = inspect.signature(optimizer_cls.__init__)
        raise PyTorchLayerArgumentsInvalidException(
            f"{layer_name} init signature is: `{signature}`, got `{layer_kwargs}`"
        ) from e


class PyTorchLayerNotFoundException(Exception):
    ...


class PyTorchLayerArgumentsInvalidException(Exception):
    ...
