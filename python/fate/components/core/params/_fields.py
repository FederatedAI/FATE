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

import typing
from typing import Any, Optional, Type, TypeVar

import pydantic


class Parameter:
    @classmethod
    def parse(cls, obj: Any):
        return pydantic.parse_obj_as(cls, obj)

    @classmethod
    def schema(cls):
        return NotImplemented


T = TypeVar("T")


class _SmartUnion(pydantic.BaseModel.Config):
    smart_union = True


def parse(type_: Type[T], obj: Any) -> T:
    if not isinstance(type_, typing._GenericAlias) and issubclass(type_, Parameter):
        return type_.parse(obj)
    else:
        # create_model to inject config
        model = pydantic.create_model("parameter", __config__=_SmartUnion, p=(type_, ...))
        return pydantic.parse_obj_as(model, {"p": obj}).p


def jsonschema(type_: Type[T]):
    return pydantic.schema_json_of(type_, indent=2)


class ConstrainedInt(pydantic.ConstrainedInt, Parameter):
    ...


def conint(
    *,
    strict: bool = False,
    gt: int = None,
    ge: int = None,
    lt: int = None,
    le: int = None,
    multiple_of: int = None,
) -> Type[int]:
    namespace = dict(strict=strict, gt=gt, ge=ge, lt=lt, le=le, multiple_of=multiple_of)
    return type("ConstrainedIntValue", (ConstrainedInt,), namespace)


class ConstrainedFloat(pydantic.ConstrainedFloat, Parameter):
    ...


def confloat(
    *,
    strict: bool = False,
    gt: float = None,
    ge: float = None,
    lt: float = None,
    le: float = None,
    multiple_of: float = None,
    allow_inf_nan: Optional[bool] = None,
) -> Type[float]:
    namespace = dict(
        strict=strict,
        gt=gt,
        ge=ge,
        lt=lt,
        le=le,
        multiple_of=multiple_of,
        allow_inf_nan=allow_inf_nan,
    )
    return type("ConstrainedFloatValue", (ConstrainedFloat,), namespace)


class StringChoice(str, Parameter):
    choice = set()
    lower = True

    @classmethod
    def __get_validators__(cls):
        yield cls.string_choice_validator

    @classmethod
    def string_choice_validator(cls, v):
        allowed = {c.lower() for c in cls.choice} if cls.lower else cls.choice
        provided = v.lower() if cls.lower else v
        if provided in allowed:
            return provided
        raise ValueError(f"provided `{provided}` not in `{allowed}`")


def string_choice(choice, lower=True) -> Type[str]:
    namespace = dict(
        choice=choice,
        lower=lower,
    )
    return type("StringChoice", (StringChoice,), namespace)
