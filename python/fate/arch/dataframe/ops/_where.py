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
from typing import List
from .._dataframe import DataFrame
from ..manager import BlockType
from ..manager import DataManager


def where(df: DataFrame, other: DataFrame):
    if df.shape[0] != other.shape[0]:
        raise ValueError("Row numbers should be identical.")

    data_manager = df.data_manager
    other_data_manager = other.data_manager

    column_names = data_manager.infer_operable_field_names()
    other_column_names = other_data_manager.infer_operable_field_names()

    if (set(column_names) & set(other_column_names)) != set(column_names):
        raise ValueError("To use df[mask], mask's columns should contains all df's columns")

    if column_names != other_column_names:
        other = other[column_names]
        other_column_names = column_names

    false_column_set = _get_false_columns(other)
    if not false_column_set:
        return df

    """
    need type promote ?
    """
    need_promoted = False
    for name in column_names:
        bid = data_manager.loc_block(name, with_offset=False)
        if not BlockType.is_float(data_manager.get_block(bid).block_type):
            need_promoted = True
            break

    if not need_promoted:
        block_table = _where_float_type(df.block_table, other.block_table,
                                        data_manager, other.data_manager, column_names)
        return DataFrame(
            df._ctx,
            block_table,
            df.partition_order_mappings,
            data_manager.duplicate()
        )


def _get_false_columns(df: DataFrame):
    block_table = df.block_table
    data_manager = df.data_manager
    block_index_set = set(data_manager.infer_operable_blocks())

    false_table = block_table.mapValues(
        lambda blocks: [
            block.all(axis=0) if bid in block_index_set else []
            for bid, block in enumerate(blocks)
        ]
    )

    false_values = false_table.reduce(
        lambda blocks1, blocks2:
        [
            block1 & block2 if bid in block_index_set else []
            for bid, (block1, block2) in enumerate(zip(blocks1, blocks2))
        ]
    )

    false_columns = set()
    column_names = data_manager.infer_operable_field_names()
    for name in column_names:
        _bid, _offset = data_manager.loc_block(name)
        if isinstance(false_values[_bid], torch.Tensor):
            if not false_values[_bid][_offset].item():
                false_columns.add(name)
        elif isinstance(false_values[_bid], np.ndarray):
            if not false_values[_bid][_offset]:
                false_columns.add(name)

    return false_columns


def _where_float_type(l_block_table, r_block_table,
                      l_data_manager: "DataManager",
                      r_data_manager: "DataManager",
                      column_names: List[str]):
    l_loc_info = [l_data_manager.loc_block(name) for name in column_names]
    r_loc_info = [r_data_manager.loc_block(name) for name in column_names]

    def __convert_na(l_blocks, r_blocks):
        ret_blocks = []
        for block in l_blocks:
            if isinstance(block, torch.Tensor):
                ret_blocks.append(block.clone())
            elif isinstance(block, np.ndarray):
                ret_blocks.append(np.copy(block))
            else:
                ret_blocks.append(block)

        for (l_bid, l_offset), (r_bid, r_offset) in zip(l_loc_info, r_loc_info):
            if isinstance(ret_blocks[l_bid], torch.Tensor):
                ret_blocks[l_bid][:, l_offset][~r_blocks[r_bid][:, r_offset]] = torch.nan
            else:
                ret_blocks[l_bid][:, l_offset][~r_blocks[r_bid][:, r_offset]] = np.nan

        return ret_blocks

    return l_block_table.join(r_block_table, __convert_na)
