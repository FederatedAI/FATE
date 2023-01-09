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
from ._ops import auto_binary_op


@auto_binary_op
def add(x, y, *args, **kwargs):
    """"""
    ...


@auto_binary_op
def sub(x, y, *args, **kwargs):
    """"""
    ...


@auto_binary_op
def mul(x, y, *args, **kwargs):
    """"""
    ...


@auto_binary_op
def div(x, y, *args, **kwargs):
    """"""
    ...


@auto_binary_op
def pow(x, y, *args, **kwargs):
    """"""
    ...


@auto_binary_op
def remainder(x, y, *args, **kwargs):
    """"""
    ...


@auto_binary_op
def fmod(x, y, *args, **kwargs):
    "element wise remainder of division"
    ...
