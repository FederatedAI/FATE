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


from ._tensor import Tensor
from .distributed import DStorage


def _unwraped_as_storage(*args, **kwargs):
    sargs = []
    skwargs = {}
    is_distributed = False
    for k in args:
        if isinstance(k, Tensor):
            if isinstance(k.storage, DStorage):
                is_distributed = True
            sargs.append(k.storage)
        else:
            sargs.append(k)
    for k, v in kwargs.items():
        if isinstance(v, Tensor):
            if isinstance(v.storage, DStorage):
                is_distributed = True
            skwargs[k] = v.storage
        else:
            skwargs[k] = v
    return is_distributed, sargs, skwargs


def _auto_dispatch(func):
    name = func.__name__

    def wraped(*args, **kwargs):
        _is_distributed, sargs, skwargs = _unwraped_as_storage(*args, **kwargs)
        if _is_distributed:
            from fate.arch.tensor.distributed import ops
        else:
            from fate.arch.storage import _ops as ops
        op = getattr(ops, name)
        if op is None:
            raise NotImplementedError(f"op `{name}` not found in {ops}")
        output_storage = op(*sargs, **skwargs)
        return Tensor(output_storage)

    return wraped
