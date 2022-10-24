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
import copy
import inspect

import torch.optim
from federatedml.nn.backend.pytorch.custom import optimizer as custom_optimizers

from federatedml.util import LOGGER


def _supported_optimizer():
    optimizers = []
    for module in [torch.optim, custom_optimizers]:
        for name, _ in inspect.getmembers(
            module,
            lambda cls: inspect.isclass(cls) and issubclass(cls, torch.optim.Optimizer),
        ):
            if name.startswith("_"):
                continue
            optimizers.append(name)

    return optimizers


def get_optimizer_cls(optimizer_name):
    modules = [torch.optim, custom_optimizers]
    for i, module in enumerate(modules):
        if hasattr(module, optimizer_name) and issubclass(
            getattr(module, optimizer_name), torch.optim.Optimizer
        ):
            return getattr(module, optimizer_name)
        LOGGER.debug(
            f"optimizer cls {optimizer_name} not found in {module}, searching {modules[i+1:]}"
        )
    raise PyTorchOptimizerNotFoundException(
        f"optimizer named `{optimizer_name}` not found, use one of `{_supported_optimizer()}`"
    )


def get_optimizer(parameters, optimizer_name, optimizer_kwargs):
    optimizer_cls = get_optimizer_cls(optimizer_name)
    kwargs = copy.deepcopy(optimizer_kwargs)
    kwargs["params"] = parameters
    try:
        return optimizer_cls(**kwargs)
    except Exception as e:
        signature = inspect.signature(optimizer_cls.__init__)
        raise PyTorchOptimizerArgumentsInvalidException(
            f"{optimizer_name} init signature is: `{signature}`, got `{kwargs}`"
        ) from e


class PyTorchOptimizerNotFoundException(Exception):
    ...


class PyTorchOptimizerArgumentsInvalidException(Exception):
    ...
