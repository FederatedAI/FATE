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
import numpy as np
from .._dataframe import DataFrame
from ._compress_block import compress_blocks


def cmp_operate(lhs: DataFrame, rhs, op) -> "DataFrame":
    data_manager = lhs.data_manager
    block_indexes = data_manager.infer_operable_blocks()
    column_names = data_manager.infer_operable_field_names()

    if isinstance(rhs, (bool, int, float, np.int32, np.float32, np.int64, np.float64, np.bool)):
        ...
    elif isinstance(rhs, list):
        ...
    elif isinstance(rhs, DataFrame):
        other_column_names = data_manager.infer_operable_field_names()
        if column_names != other_column_names:
            ...
