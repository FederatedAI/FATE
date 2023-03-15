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
import functools

from .._dataframe import DataFrame
from ..manager import DataManager
from ..manager.block_manager import BlockType


def sigmoid(df: "DataFrame") -> "DataFrame":
    data_manager = df.data_manager.duplicate()
    operable_blocks = data_manager.infer_operable_blocks()
    non_operable_blocks = data_manager.infer_non_operable_blocks()
    for block_id in operable_blocks:
        if not data_manager.blocks[block_id].is_numeric():
            raise ValueError("Sigmoid support only operates on numeric columns")
        if data_manager.blocks[block_id].block_type in [BlockType.int32, BlockType.int64]:
            data_manager.blocks[block_id] = data_manager.blocks[block_id].convert_block_type(BlockType.float32)

    def _sigmoid(blocks, op_blocks=None, reserved_blocks=None):
        ret_blocks = [[] for i in range(len(op_blocks) + len(reserved_blocks))]
        for bid in reserved_blocks:
            ret_blocks[bid] = blocks[bid]

        for bid in op_blocks:
            ret_blocks[bid] = blocks[bid].sigmoid()

        return ret_blocks

    _sigmoid_func = functools.partial(_sigmoid, op_blocks=operable_blocks, reserved_blocks=non_operable_blocks)

    block_table = df.block_table.mapValues(_sigmoid_func)

    return DataFrame(
        df._ctx,
        block_table,
        df.partition_order_mappings,
        data_manager
    )
