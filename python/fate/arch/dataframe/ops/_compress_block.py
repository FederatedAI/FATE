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
import numpy as np
import torch
from ..manager import BlockType
from ..manager import DataManager
from ..conf.default_config import BLOCK_COMPRESS_THRESHOLD


def compress_blocks(block_table, data_manager: DataManager, force_compress=False):
    compressed_data_manager = data_manager.duplicate()
    to_compress_block_loc, non_compress_block_changes = compressed_data_manager.compress_blocks()

    compress_block_size = 0
    for _, block_loc in to_compress_block_loc:
        compress_block_size += len(block_loc)

    if not to_compress_block_loc or (not force_compress and compress_block_size <= BLOCK_COMPRESS_THRESHOLD):
        return block_table, data_manager

    def _compress(blocks):
        ret_blocks = [[] for _ in range(compressed_data_manager.block_num)]
        for src_bid, dst_bid in non_compress_block_changes.items():
            ret_blocks[dst_bid] = blocks[src_bid]

        lines = len(blocks[0])
        for dst_bid, block_loc in to_compress_block_loc:
            block = compressed_data_manager.get_block(dst_bid)
            field_len = len(block.field_indexes)
            # TODO: empty block create logic should move to block_manager later,
            #  we pull it here as block_manager has more type like phe_tensor/pd.Index, which should not be considered in compressing
            if BlockType.is_tensor(block.block_type):
                block_buf = np.empty((lines, field_len), dtype=getattr(np, block.block_type.value))
            else:
                block_buf = np.empty((lines, field_len), dtype=object)

            for src_bid, field_indexes in block_loc:
                block_buf[:, field_indexes] = blocks[src_bid]

            if isinstance(block_buf, np.ndarray):
                ret_blocks[dst_bid] = torch.from_numpy(block_buf)

        return ret_blocks

    block_table = block_table.mapValues(_compress)

    return block_table, compressed_data_manager
