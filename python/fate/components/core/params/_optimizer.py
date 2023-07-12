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
