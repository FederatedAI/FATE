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

from typing import Type

import pydantic

from ._fields import string_choice
from ._penalty import penalty_param


class OptimizerParam(pydantic.BaseModel):
    method: string_choice(
        ["sgd", "adadelta", "adagrad", "adam", "adamax", "adamw", "asgd", "nadam", "radam", "rmsprop", "rprop"]
    ) = "sgd"
    penalty: penalty_param(l1=True, l2=True, none=True) = "l2"
    alpha: float = 1.0
    optimizer_params: dict


def optimizer_param() -> Type[OptimizerParam]:
    namespace = {}
    return type("OptimizerParam", (OptimizerParam,), namespace)
