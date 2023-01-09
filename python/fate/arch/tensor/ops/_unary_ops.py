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
from ._ops import auto_unary_op


@auto_unary_op
def abs(x, *args, **kwargs):
    "arc cosine"
    ...


@auto_unary_op
def asin(x, *args, **kwargs):
    "arc sin"
    ...


@auto_unary_op
def atan(x, *args, **kwargs):
    """"""
    ...


@auto_unary_op
def atan2(x, *args, **kwargs):
    """"""
    ...


@auto_unary_op
def ceil(x, *args, **kwargs):
    "ceiling"
    ...


@auto_unary_op
def cos(x, *args, **kwargs):
    """"""
    ...


@auto_unary_op
def cosh(x, *args, **kwargs):
    """"""
    ...


@auto_unary_op
def erf(x, *args, **kwargs):
    "Gaussian error functiom"
    ...


@auto_unary_op
def erfinv(x, *args, **kwargs):
    "Gaussian error functiom"
    ...


@auto_unary_op
def exp(x, *args, **kwargs):
    """"""
    ...


@auto_unary_op
def expm1(x, *args, **kwargs):
    "exponential of each element minus 1"
    ...


@auto_unary_op
def floor(x, *args, **kwargs):
    """"""
    ...


@auto_unary_op
def frac(x, *args, **kwargs):
    "fraction part 3.4 -> 0.4"
    ...


@auto_unary_op
def log(x, *args, **kwargs):
    "natural log"
    ...


@auto_unary_op
def log1p(x, *args, **kwargs):
    "y = log(1 + x)"
    ...


@auto_unary_op
def neg(x, *args, **kwargs):
    """"""
    ...


@auto_unary_op
def reciprocal(x, *args, **kwargs):
    "1/x"
    ...


@auto_unary_op
def sigmoid(x, *args, **kwargs):
    "sigmode(x)"
    ...


@auto_unary_op
def sign(x, *args, **kwargs):
    """"""
    ...


@auto_unary_op
def sin(x, *args, **kwargs):
    """"""
    ...


@auto_unary_op
def sinh(x, *args, **kwargs):
    """"""
    ...


@auto_unary_op
def sqrt(x, *args, **kwargs):
    """"""
    ...


@auto_unary_op
def square(x, *args, **kwargs):
    """"""
    ...


@auto_unary_op
def tan(x, *args, **kwargs):
    """"""
    ...


@auto_unary_op
def tanh(x, *args, **kwargs):
    """"""
    ...


@auto_unary_op
def trunc(x, *args, **kwargs):
    "truncated integer"
    ...


@auto_unary_op
def rsqrt(x, *args, **kwargs):
    "the reciprocal of the square-root"
    ...


@auto_unary_op
def round(x, *args, **kwargs):
    """"""
    ...
