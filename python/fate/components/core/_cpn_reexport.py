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
from typing import Callable, List, Optional, Type, TypeVar

# re-export
from .component_desc import (
    component,
    data_directory_input,
    data_directory_inputs,
    data_directory_output,
    data_directory_outputs,
    dataframe_input,
    dataframe_inputs,
    dataframe_output,
    dataframe_outputs,
    data_unresolved_output,
    data_unresolved_outputs,
    json_model_input,
    json_model_inputs,
    json_model_output,
    json_model_outputs,
    model_directory_input,
    model_directory_inputs,
    model_directory_output,
    model_directory_outputs,
    parameter,
    table_input,
    table_inputs,
    model_unresolved_output,
    model_unresolved_outputs,
)
from .essential import Role

T1 = TypeVar("T1")
T2 = TypeVar("T2")


def union(f1: Callable[..., Type[T1]], f2: Callable[..., Type[T2]]):
    def wrapper(roles: Optional[List[Role]] = None, desc="", optional=False) -> "Type[T1] | Type[T2]":
        return f1(roles, desc, optional) | f2(optional=optional)

    return wrapper


__all__ = [
    "component",
    "parameter",
    "dataframe_input",
    "dataframe_output",
    "dataframe_inputs",
    "dataframe_outputs",
    "table_input",
    "table_inputs",
    "data_directory_input",
    "data_directory_output",
    "data_directory_outputs",
    "data_directory_inputs",
    "data_unresolved_output",
    "data_unresolved_outputs",
    "json_model_output",
    "json_model_outputs",
    "json_model_input",
    "json_model_inputs",
    "model_directory_inputs",
    "model_directory_outputs",
    "model_directory_output",
    "model_directory_input",
    "model_unresolved_output",
    "model_unresolved_outputs",
]
