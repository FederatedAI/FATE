from typing import Union

import pydantic

from ._fields import string_choice


class InitParam(pydantic.BaseModel):
    method: string_choice(['zeros', 'ones', 'consts', 'random', 'random_uniform']) = 'zeros'
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
