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
from typing import List, Tuple
from ..manager.block_manager import Block, BlockType


def select_column_value(src_block_table, select_block_table,
                        target_block_id: Tuple[int, int], keeping_blocks: List[int],
                        schema_manager, block_manager):

    def _select_column(l_value, r_value):
        block_type = BlockType.bool
        select_columns = []
        bid, offset = target_block_id
        for i in range(r_value[bid].shape[0]):
            target_column = r_value[bid][offset]

            dst_column_name = schema_manager.schema.columns[target_column]
            src_bid, src_offset = block_manager.get_block_id(schema_manager.get_column_offset(dst_column_name))
            select_columns.append(l_value[src_bid][[src_offset]])

            block_type = BlockType.promote_types(block_type, l_value[src_bid].block_type)

        ret_blocks = []
        for bid in keeping_blocks:
            ret_blocks.append(l_value[bid])

        ret_blocks.append(Block.get_block_by_type(block_type)(select_columns))

        return select_columns

    return src_block_table.join(select_block_table).mapValues(_select_column)
