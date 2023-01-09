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

from fate.arch.tensor.types import LStorage, dtype


def _ops_dispatch_signature_1_local_cpu_unknown(
    method,
    dtype: dtype,
    args,
    kwargs,
) -> Callable[[LStorage], LStorage]:
    if dtype.is_basic():
        from .plain import _TorchStorage

        return _TorchStorage.unary(method, args, kwargs)
    elif dtype.is_paillier():
        from .paillier import _ops_dispatch_signature_1_local_cpu_paillier

        return _ops_dispatch_signature_1_local_cpu_paillier(method, args, kwargs)


def _ops_dispatch_signature_2_local_cpu_unknown(method, dtype: dtype, args, kwargs) -> Callable[[Any, Any], LStorage]:
    if dtype.is_basic():
        from .plain import _TorchStorage

        return _TorchStorage.binary(method, args, kwargs)
    elif dtype.is_paillier():
        from .paillier import _ops_dispatch_signature_2_local_cpu_paillier

        return _ops_dispatch_signature_2_local_cpu_paillier(method, args, kwargs)
