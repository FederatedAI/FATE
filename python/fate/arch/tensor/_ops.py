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


from ._dispatch import _auto_dispatch


@_auto_dispatch
def add(a, b, *args, **kwargs):
    ...


@_auto_dispatch
def sub(a, b, *args, **kwargs):
    ...


@_auto_dispatch
def mul(a, b, *args, **kwargs):
    ...


@_auto_dispatch
def div(a, b, *args, **kwargs):
    ...


@_auto_dispatch
def truediv(a, b, *args, **kwargs):
    ...


@_auto_dispatch
def pow(a, b, *args, **kwargs):
    ...


@_auto_dispatch
def remainder(a, b, *args, **kwargs):
    ...


@_auto_dispatch
def fmod(a, b, *args, **kwargs):
    ...


@_auto_dispatch
def sum(a, *args, **kwargs):
    ...


@_auto_dispatch
def mean(a, *args, **kwargs):
    ...


@_auto_dispatch
def std(a, *args, **kwargs):
    ...


@_auto_dispatch
def var(a, *args, **kwargs):
    ...


@_auto_dispatch
def max(a, *args, **kwargs):
    ...


@_auto_dispatch
def min(a, *args, **kwargs):
    ...


@_auto_dispatch
def abs(a, *args, **kwargs):
    ...


@_auto_dispatch
def asin(a, *args, **kwargs):
    ...


@_auto_dispatch
def atan(a, *args, **kwargs):
    ...


@_auto_dispatch
def atan2(a, *args, **kwargs):
    ...


@_auto_dispatch
def ceil(a, *args, **kwargs):
    ...


@_auto_dispatch
def cos(a, *args, **kwargs):
    ...


@_auto_dispatch
def cosh(a, *args, **kwargs):
    ...


@_auto_dispatch
def erf(a, *args, **kwargs):
    ...


@_auto_dispatch
def erfinv(a, *args, **kwargs):
    ...


@_auto_dispatch
def exp(a, *args, **kwargs):
    ...


@_auto_dispatch
def expm1(a, *args, **kwargs):
    ...


@_auto_dispatch
def floor(a, *args, **kwargs):
    ...


@_auto_dispatch
def frac(a, *args, **kwargs):
    ...


@_auto_dispatch
def log(a, *args, **kwargs):
    ...


@_auto_dispatch
def log1p(a, *args, **kwargs):
    ...


@_auto_dispatch
def neg(a, *args, **kwargs):
    ...


@_auto_dispatch
def reciprocal(a, *args, **kwargs):
    ...


@_auto_dispatch
def sigmoid(a, *args, **kwargs):
    ...


@_auto_dispatch
def sign(a, *args, **kwargs):
    ...


@_auto_dispatch
def sin(a, *args, **kwargs):
    ...


@_auto_dispatch
def sinh(a, *args, **kwargs):
    ...


@_auto_dispatch
def sqrt(a, *args, **kwargs):
    ...


@_auto_dispatch
def square(a, *args, **kwargs):
    ...


@_auto_dispatch
def tan(a, *args, **kwargs):
    ...


@_auto_dispatch
def tanh(a, *args, **kwargs):
    ...


@_auto_dispatch
def trunc(a, *args, **kwargs):
    ...


@_auto_dispatch
def rsqrt(a, *args, **kwargs):
    ...


@_auto_dispatch
def round(a, *args, **kwargs):
    ...


@_auto_dispatch
def slice(a, key):
    ...


@_auto_dispatch
def matmul(a, b):
    ...


@_auto_dispatch
def quantile(a, q, epsilon):
    ...


@_auto_dispatch
def encrypt(a, encryptor):
    ...


@_auto_dispatch
def decrypt(a, decryptor):
    ...
