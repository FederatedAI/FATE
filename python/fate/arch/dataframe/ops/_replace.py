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
import functools
import numpy as np
import torch
from ._compress_block import compress_blocks
from .._dataframe import DataFrame
from ..manager import BlockType, DataManager


def replace(df: "DataFrame", to_replace: dict):
    data_manager = df.data_manager.duplicate()
    field_names = list(filter(lambda field_name: field_name in to_replace, data_manager.infer_operable_field_names()))
    blocks_loc = data_manager.loc_block(field_names)

    dst_block_types = []
    _to_replace_list = []
    for name, (_bid, _) in zip(field_names, blocks_loc):
        block_type = data_manager.get_block(_bid).block_type
        for k, v in to_replace[name].items():
            v_type = BlockType.get_block_type(v)

            if block_type < v_type:
                block_type = v_type

        dst_block_types.append(block_type)
        _to_replace_list.append((_bid, _, to_replace[name]))

    narrow_blocks, dst_blocks = data_manager.split_columns(field_names, dst_block_types)

    def _mapper(blocks, to_replace_list: list = None, narrow_loc: list = None,
                dst_bids: list = None, dm: DataManager = None):
        ret_blocks = []
        for block in blocks:
            if isinstance(block, torch.Tensor):
                ret_blocks.append(block.clone())
            elif isinstance(block, np.ndarray):
                ret_blocks.append(block.copy())
            else:
                ret_blocks.append(block)

        for i in range(len(ret_blocks), dm.block_num):
            ret_blocks.append([])

        for bid, offsets in narrow_loc:
            ret_blocks[bid] = ret_blocks[bid][:, offsets]

        for dst_bid, (src_bid, src_offset, _to_replace_dict) in zip(dst_bids, to_replace_list):
            row_values = blocks[src_bid][:, src_offset]
            replace_ret = []
            is_torch = torch.is_tensor(row_values)
            for idx, value in enumerate(row_values):
                if is_torch:
                    value = value.item()
                if value not in _to_replace_dict:
                    replace_ret.append([value])
                else:
                    replace_ret.append([_to_replace_dict[value]])

            ret_blocks[dst_bid] = dm.blocks[dst_bid].convert_block(replace_ret)

        return ret_blocks

    replace_mapper = functools.partial(_mapper,
                                       to_replace_list=_to_replace_list,
                                       narrow_loc=narrow_blocks,
                                       dst_bids=dst_blocks,
                                       dm=data_manager)

    block_table = df.block_table.mapValues(replace_mapper)

    block_indexes = data_manager.infer_operable_blocks()
    if len(block_indexes) > 1:
        to_promote_types = []
        for _bid in block_indexes:
            to_promote_types.append((_bid, data_manager.get_block(_bid).block_type))

        data_manager.promote_types(to_promote_types)
        block_table, data_manager = compress_blocks(block_table, data_manager)

    return DataFrame(
        df._ctx,
        block_table,
        partition_order_mappings=df.partition_order_mappings,
        data_manager=data_manager
    )
