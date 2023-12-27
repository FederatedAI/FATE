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

from typing import Union

import pydantic

from ._fields import string_choice


class InitParam(pydantic.BaseModel):
    method: string_choice(["zeros", "ones", "consts", "random", "random_uniform"]) = "zeros"
    fill_val: Union[int, float] = 0.0
    fit_intercept: bool = True
    random_state: int = None


def init_param():
    namespace = {}
    return type("InitParam", (InitParam,), namespace)


"""
class OnesInitParam(pydantic.BaseModel):
    method: Literal['ones']
    fit_intercept: bool = True


class ConstsInitParam(pydantic.BaseModel):
    method: Literal['consts']
    fill_val: Union[int, float]
    fit_intercept: bool = True


class RandomInitParam(pydantic.BaseModel):
    method: Literal['random']
    fit_intercept: bool = True


InitParam = Union[ZerosInitParam, OnesInitParam, ConstsInitParam, RandomInitParam]

"""
