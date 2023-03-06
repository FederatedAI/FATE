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
import pandas as pd
from typing import List, Tuple
import torch
from fate.arch import tensor


def transform_to_tensor(ctx, block_table,
                        block_indexes: List[Tuple[int, int]], retrieval_block_indexes, dtype=None):
    """
    column_indexes: column to retrieval
    block_indexes: list, (block_id, block_indexes)
    retrieval_block_indexes: list, each element: (src_block_id, dst_block_id, changed=True/False, block_indexes)
    dtype: convert to tensor with dtype, default is None
    """
    def _to_local_tensor(src_blocks):
        if len(retrieval_block_indexes) == 1:
            src_block_id, dst_block_id, is_changed, indexes = retrieval_block_indexes[0]
            if not is_changed:
                t = src_blocks[src_block_id]
            else:
                t = src_blocks[src_block_id][:, indexes]
        else:
            i = 0
            tensors = []
            while i < len(block_indexes):
                bid = block_indexes[i][0]
                indexes = [block_indexes[i][1]]
                j = i + 1
                while j < len(block_indexes) and block_indexes[j] == block_indexes[j - 1]:
                    indexes.append(block_indexes[j][1])
                    j += 1

                tensors.append(src_blocks[bid][:, indexes])
                i = j

            t = torch.hstack(tensors)

        if dtype:
            t = t.type(getattr(torch, t))

        return t

    local_tensor_table = block_table.mapValues(_to_local_tensor)
    local_tensor_blocks = [block_with_id[1] for block_with_id in sorted(local_tensor_table.collect())]

    return tensor.distributed_tensor(ctx,
                                     local_tensor_blocks,
                                     partitions=len(local_tensor_blocks))


def transform_block_to_list(block_table, block_manager):
    column_block_mapping = block_manager.column_block_mapping
    block_indexes = [[]] * len(column_block_mapping)
    for col_id, _block_id_tuple in column_block_mapping.items():
        block_indexes[col_id] = _block_id_tuple

    def _to_list(src_blocks):
        i = 0
        dst_list = None
        lines = 0
        while i < len(block_indexes):
            bid = block_indexes[i][0]
            if isinstance(src_blocks[bid], pd.Index):
                if not dst_list:
                    lines = len(src_blocks[bid])
                    dst_list = [[] for i in range(lines)]

                for j in range(lines):
                    dst_list[j].append(src_blocks[bid][j])

                if len(dst_list[0]) > 111:
                    assert 1 == 2, (i, bid, src_blocks[bid], src_blocks[bid][0], dst_list[0])
                i += 1
            else:
                """
                pd.values or tensor
                """
                indexes = [block_indexes[i][1]]
                j = i + 1
                while j < len(block_indexes) and block_indexes[j] == block_indexes[j - 1]:
                    indexes.append(block_indexes[j][1])
                    j += 1

                if isinstance(src_blocks[bid], pd.DataFrame):
                    for line_id, row_value in enumerate(src_blocks[bid].values[:, indexes]):
                        dst_list[line_id].extend(row_value)
                else:
                    for line_id, row_value in enumerate(src_blocks[bid].to_local()[:, indexes].tolist()):
                        dst_list[line_id].extend(row_value)

                i = j

        return dst_list

    return block_table.mapValues(_to_list)


def transform_list_to_block(table, block_manager):
    from ..manager.block_manager import BlockType

    def _to_block(values):
        convert_blocks = []

        lines = len(values)
        for block_schema in block_manager.blocks:
            if block_schema.block_type == BlockType.index and len(block_schema.column_indexes) == 1:
                col_idx = block_schema.column_indexes[0]
                block_content = [values[i][col_idx] for i in range(lines)]
            else:
                block_content = []
                for i in range(lines):
                    buf = []
                    for col_idx in block_schema.column_indexes:
                        buf.append(values[i][col_idx])
                    block_content.append(buf)

            convert_blocks.append(block_schema.convert_block(block_content))

        return convert_blocks

    return table.mapValues(_to_block)


def transform_list_block_to_frame_block(block_table, block_manager):
    def _to_frame_block(blocks):
        convert_blocks = []
        for idx, block_schema in enumerate(block_manager.blocks):
            block_content = [block[idx] for block in blocks]
            try:
                block_schema.convert_block(block_content)
            except:
                assert 1 == 2, block_content
            convert_blocks.append(block_schema.convert_block(block_content))

        return convert_blocks

    return block_table.mapValues(_to_frame_block)
