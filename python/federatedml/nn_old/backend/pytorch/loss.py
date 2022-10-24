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

import numpy as np
import torch

# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss

from federatedml.nn.backend.pytorch.custom import loss as custom_losses
from federatedml.util import LOGGER


def _supported_losses():
    losses = []
    for module in [torch.nn.modules.loss, custom_losses]:
        for name, _ in inspect.getmembers(
            module, lambda cls: inspect.isclass(cls) and issubclass(cls, _Loss)
        ):
            if name.startswith("_"):
                continue
            losses.append(name)
    return losses


def get_loss_cls(loss_name):
    modules = [torch.nn.modules.loss, custom_losses]
    for i, module in enumerate(modules):
        if hasattr(module, loss_name) and issubclass(getattr(module, loss_name), _Loss):
            return getattr(module, loss_name)
        LOGGER.debug(
            f"optimizer cls {loss_name} not found in {module}, searching {modules[i + 1:]}"
        )

    raise PyTorchLossNotFoundException(
        f"loss named `{loss_name}` not found, use one of {_supported_losses()}"
    )


def get_loss_fn(loss_name, loss_kwargs):
    loss_cls = get_loss_cls(loss_name)
    try:
        loss_fn = loss_cls(**loss_kwargs)
        if isinstance(loss_fn, (torch.nn.CrossEntropyLoss, torch.nn.NLLLoss)):
            return loss_fn, np.int64
        else:
            return loss_fn, np.float32
    except TypeError as e:
        signature = inspect.signature(loss_cls.__init__)
        raise PyTorchLossArgumentsInvalidException(
            f"{loss_name} init signature is: {signature}, got {loss_kwargs}"
        ) from e


class PyTorchLossNotFoundException(Exception):
    ...


class PyTorchLossArgumentsInvalidException(Exception):
    ...
