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
#
import operator
from .._dataframe import DataFrame
from ..manager import BlockType
from .utils.operators import unary_operate


def invert(df: DataFrame):
    data_manager = df.data_manager
    block_indexes = data_manager.infer_operable_blocks()
    for bid in block_indexes:
        if data_manager.blocks[bid] != BlockType.bool:
            raise ValueError("to use ~df syntax, data types should be bool")

    block_table = unary_operate(df.block_table, operator.invert, block_indexes)
    return type(df)(df.ctx, block_table, df.partition_order_mappings, data_manager.duplicate())
