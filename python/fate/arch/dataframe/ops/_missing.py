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
import torch
from ._compress_block import compress_blocks
from .._dataframe import DataFrame
from ..manager import BlockType


def isna(df: "DataFrame"):
    data_manager = df.data_manager
    block_indexes = data_manager.infer_operable_blocks()

    block_table = _isna(df.block_table, block_indexes)
    dst_data_manager = data_manager.duplicate()
    to_promote_types = []
    for bid in block_indexes:
        to_promote_types.append((bid, BlockType.get_block_type(torch.bool)))

    dst_data_manager.promote_types(to_promote_types)
    dst_block_table, dst_data_manager = compress_blocks(block_table, dst_data_manager)

    return DataFrame(
        df._ctx,
        dst_block_table,
        df.partition_order_mappings,
        dst_data_manager
    )


def _isna(block_table, block_indexes):
    block_index_set = set(block_indexes)

    def _isna_judgement(blocks):
        ret_blocks = []
        for bid, block in enumerate(blocks):
            if bid not in block_index_set:
                ret_blocks.append(block)
            else:
                ret_blocks.append(torch.isnan(block) if isinstance(block, torch.Tensor) else np.isnan(block))

        return ret_blocks

    return block_table.mapValues(
        _isna_judgement
    )
