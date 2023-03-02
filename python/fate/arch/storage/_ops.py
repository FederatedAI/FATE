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

"""
This file contains operations on `Storage` used for Tensor and Dataframe data.
Most operations are simply wrapped and dispatched to concrete Storage.
We have provided this wrapper for a better development experience:
using explicit function calls like min(...) over get_ops('min')(...) is recommended.

```
from fate.arch import storage

storage.max(...)
```
"""

from ._dispatch import _auto_dispatch, _auto_dispatch_decrypt, _auto_dispatch_encrypt


@_auto_dispatch
def exp(s, *args, **kwargs):
    ...


@_auto_dispatch
def std(s, *args, **kwargs):
    ...


@_auto_dispatch
def var(s, *args, **kwargs):
    ...


@_auto_dispatch
def sum(s, *args, **kwargs):
    ...


@_auto_dispatch
def square(s, *args, **kwargs):
    ...


@_auto_dispatch
def sqrt(s, *args, **kwargs):
    ...


@_auto_dispatch
def min(s, *args, **kwargs):
    ...


@_auto_dispatch
def max(s, *args, **kwargs):
    ...


@_auto_dispatch
def minimum(s1, s2, *args, **kwargs):
    ...


@_auto_dispatch
def maximum(s1, s2, *args, **kwargs):
    ...


@_auto_dispatch
def add(s1, s2, *args, **kwargs):
    ...


@_auto_dispatch
def sub(s1, s2, *args, **kwargs):
    ...


@_auto_dispatch
def mul(s1, s2, *args, **kwargs):
    ...


@_auto_dispatch
def div(s1, s2, *args, **kwargs):
    ...


@_auto_dispatch
def truediv(s1, s2, *args, **kwargs):
    ...


@_auto_dispatch
def slice(s, *args, **kwargs):
    ...


@_auto_dispatch_encrypt
def encrypt(s, encryptor):
    ...


@_auto_dispatch_decrypt
def decrypt(s, decryptor):
    ...


@_auto_dispatch
def quantile(s, q, epsilon):
    ...


@_auto_dispatch
def quantile_summary(s, epsilon):
    ...


@_auto_dispatch
def matmul(s1, s2, *args, **kwargs):
    ...
