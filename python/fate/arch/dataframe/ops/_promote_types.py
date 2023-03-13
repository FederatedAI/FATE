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
from ..manager import DataManager
from ..manager.block_manager import Block


def promote_types(block_table, data_manager: DataManager, to_promote_blocks):
    data_manager.promote_types(to_promote_blocks)
    to_promote_block_dict = dict((bid, block_type) for bid, block_type in to_promote_blocks)
    block_table = block_table.mapValues(
        lambda blocks: [
            blocks[bid] if bid not in to_promote_block_dict
            else Block.get_block_by_type(to_promote_block_dict[bid]).convert_block(blocks[bid].tolist())
            for bid in range(len(blocks))
        ]
    )

    return block_table, data_manager
