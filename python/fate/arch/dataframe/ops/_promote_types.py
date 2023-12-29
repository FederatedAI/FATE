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
import torch
from ..manager import DataManager
from ..manager.block_manager import Block
from typing import List, Tuple


def promote_types(block_table, data_manager: DataManager, to_promote_blocks):
    data_manager.promote_types(to_promote_blocks)
    to_promote_block_dict = dict((bid, block_type) for bid, block_type in to_promote_blocks)
    block_table = block_table.mapValues(
        lambda blocks: [
            blocks[bid]
            if bid not in to_promote_block_dict
            else Block.get_block_by_type(to_promote_block_dict[bid]).convert_block(blocks[bid].tolist())
            for bid in range(len(blocks))
        ]
    )

    return block_table, data_manager


def promote_partial_block_types(
    block_table, narrow_blocks, dst_blocks, dst_fields_loc, data_manager: DataManager, inplace=True
):
    def _mapper(
        blocks,
        narrow_loc: list = None,
        dst_bids: list = None,
        dst_loc: List[Tuple[str, str]] = None,
        dm: DataManager = None,
        inp: bool = True,
    ):
        ret_blocks = []
        for block in blocks:
            if inp:
                if isinstance(block, torch.Tensor):
                    ret_blocks.append(block.clone())
                else:
                    ret_blocks.append(block.copy())
            else:
                ret_blocks.append(block)

        for i in range(len(ret_blocks), dm.block_num):
            ret_blocks.append([])

        for bid, offsets in narrow_loc:
            ret_blocks[bid] = ret_blocks[bid][:, offsets]

        for dst_bid, (src_bid, src_offset) in zip(dst_bids, dst_loc):
            block_values = blocks[src_bid][:, [src_offset]]
            ret_blocks[dst_bid] = dm.blocks[dst_bid].convert_block(block_values)

        return ret_blocks

    _mapper_func = functools.partial(
        _mapper, narrow_loc=narrow_blocks, dst_bids=dst_blocks, dst_loc=dst_fields_loc, dm=data_manager, inp=inplace
    )

    return block_table.mapValues(_mapper_func)
