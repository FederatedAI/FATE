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
import typing
from typing import Union


from .._dataframe import DataFrame
from ..manager import BlockType, DataManager
from ._compress_block import compress_blocks

if typing.TYPE_CHECKING:
    from fate.arch.histogram import DistributedHistogram, HistogramBuilder


def distributed_hist_stat(
    df: DataFrame, histogram_builder: "HistogramBuilder", position: DataFrame, targets: Union[dict, DataFrame]
) -> "DistributedHistogram":
    block_table, data_manager = _try_to_compress_table(df.block_table, df.data_manager, force_compress=True)
    data_block_id = data_manager.infer_operable_blocks()[0]
    position_block_id = position.data_manager.infer_operable_blocks()[0]

    if isinstance(targets, dict):

        def _pack_data_with_position(l_blocks, r_blocks, l_block_id=None, r_block_id=None):
            return l_blocks[l_block_id], r_blocks[r_block_id], dict()

        def _pack_with_target(l_values, r_value, target_name):
            l_values[2][target_name] = r_value

            return l_values

        _pack_func = functools.partial(
            _pack_data_with_position, l_block_id=data_block_id, r_block_id=position_block_id
        )

        data_with_position = block_table.join(position.block_table, _pack_func)

        for name, target in targets.items():
            _pack_with_target_func = functools.partial(_pack_with_target, target_name=name)
            data_with_position = data_with_position.join(target.shardings._data, _pack_with_target_func)
    else:
        data_with_position = block_table.join(
            position.block_table, lambda l_blocks, r_blocks: (l_blocks[data_block_id], r_blocks[position_block_id])
        )

        target_data_manager = targets.data_manager
        target_field_names = target_data_manager.infer_operable_field_names()
        fields_loc = target_data_manager.loc_block(target_field_names, with_offset=True)

        def _pack_with_targets(l_blocks, r_blocks):
            target_blocks = dict()
            for field_name, (block_id, offset) in zip(target_field_names, fields_loc):
                if (block := target_data_manager.get_block(block_id)).is_phe_tensor():
                    target_blocks[field_name] = block.convert_to_phe_tensor(
                        r_blocks[block_id], shape=(len(r_blocks[0]), 1)
                    )
                else:
                    target_blocks[field_name] = r_blocks[block_id][:, [offset]]

            return l_blocks[0], l_blocks[1], target_blocks

        data_with_position = data_with_position.join(targets.block_table, _pack_with_targets)

    return histogram_builder.statistic(data_with_position)


def _try_to_compress_table(block_table, data_manager: DataManager, force_compress=False):
    block_indexes = data_manager.infer_operable_blocks()
    if len(block_indexes) == 1:
        return block_table, data_manager

    block_type = None
    for block_id in block_indexes:
        _type = data_manager.get_block(block_id).block_type
        if not BlockType.is_integer(_type):
            raise ValueError("To use hist interface, indexes type should be integer >= 0")

        if not block_type:
            block_type = _type
        elif block_type < _type:
            block_type = _type

    to_promote_types = []
    for bid in block_indexes:
        to_promote_types.append((bid, block_type))

    data_manager.promote_types(to_promote_types)
    block_table, data_manager = compress_blocks(block_table, data_manager, force_compress=force_compress)

    return block_table, data_manager
