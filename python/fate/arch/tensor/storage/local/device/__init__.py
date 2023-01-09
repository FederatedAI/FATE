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
from typing import Any, Callable

from fate.arch.unify import device

from ....types import LStorage


def _ops_dispatch_signature1_local_unknown_unknown(
    method,
    _device,
    dtype,
    args,
    kwargs,
) -> Callable[[LStorage], LStorage]:
    if _device == device.CPU:
        from .cpu._base import _ops_dispatch_signature_1_local_cpu_unknown

        return _ops_dispatch_signature_1_local_cpu_unknown(method, dtype, args, kwargs)
    raise ValueError()


def _ops_dispatch_signature2_local_unknown_unknown(
    method, _device, dtype, args, kwargs
) -> Callable[[Any, Any], LStorage]:
    if _device == device.CPU:
        from .cpu._base import _ops_dispatch_signature_2_local_cpu_unknown

        return _ops_dispatch_signature_2_local_cpu_unknown(method, dtype, args, kwargs)
    raise ValueError()
