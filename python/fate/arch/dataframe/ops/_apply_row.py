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
import pandas as pd

from collections import Iterable

from .._dataframe import DataFrame
from ..manager.block_manager import Block, BlockType
from ..manager.data_manager import DataManager
from ..utils._auto_column_name_generated import generated_default_column_names


def apply_row(df: "DataFrame", func,
              columns: list=None, with_label=False, with_weight=False,
              enable_type_align_checking=True) -> "DataFrame":
    """
    In current version, assume that the apply_row results' lengths are equal
    """
    data_manager = df.data_manager
    dst_data_manager, _ = data_manager.derive_new_data_manager(with_sample_id=True,
                                                               with_match_id=True,
                                                               with_label=not with_weight,
                                                               with_weight=not with_weight,
                                                               columns=None)

    non_operable_field_names = dst_data_manager.get_field_name_list()
    non_operable_blocks = [data_manager.loc_block(field_name,
                                                  with_offset=False) for field_name in non_operable_field_names]

    fields_loc = data_manager.get_fields_loc(with_sample_id=False, with_match_id=False,
                                             with_label=with_label, with_weight=with_weight)

    fields_name = data_manager.get_field_name_list(with_sample_id=False,
                                                   with_match_id=False,
                                                   with_label=with_label,
                                                   with_weight=with_weight)

    _apply_func = functools.partial(_apply, func=func, src_field_names=fields_name,
                                    src_fields_loc=fields_loc, src_non_operable_blocks=non_operable_blocks,
                                    ret_columns=columns, dst_dm=dst_data_manager,
                                    enable_type_align_checking=enable_type_align_checking)

    dst_block_table_with_dm = df.block_table.mapValues(_apply_func)
    dst_data_manager = dst_block_table_with_dm.first()[1][1]
    dst_block_table = dst_block_table_with_dm.mapValues(lambda blocks_with_dm: blocks_with_dm[0])

    return DataFrame(
        df._ctx,
        dst_block_table,
        df.partition_order_mappings,
        dst_data_manager
    )


def _apply(blocks, func=None, src_field_names=None,
           src_fields_loc=None, src_non_operable_blocks=None, ret_columns=None,
           dst_dm: "DataManager"=None, enable_type_align_checking=True):
    dm = dst_dm.duplicate()
    apply_blocks = []
    lines = len(blocks[0])
    ret_column_len = len(ret_columns) if ret_columns is not None else None
    block_types = []

    flat_blocks = [Block.transform_block_to_list(block) for block in blocks]
    for lid in range(lines):
        apply_row_data = [flat_blocks[bid][lid][offset] for bid, offset in src_fields_loc]
        apply_ret = func(pd.Series(apply_row_data, index=src_field_names))

        if isinstance(apply_ret, Iterable):
            apply_ret = list(apply_ret)
            if ret_column_len is None:
                ret_column_len = len(apply_ret)
            elif ret_column_len != len(apply_ret):
                raise ValueError("Result of apply row should have equal length")
        else:
            if ret_column_len and ret_column_len != 1:
                raise ValueError("Result of apply row should have equal length")
            apply_ret = [apply_ret]

        if ret_column_len is None:
            ret_column_len = len(apply_ret)

        if not block_types:
            block_types = [BlockType.get_block_type(value) for value in apply_ret]
            apply_blocks = [[] for _ in range(ret_column_len)]

        for idx, value in enumerate(apply_ret):
            apply_blocks[idx].append([value])

        if enable_type_align_checking:
            for idx, value in enumerate(apply_ret):
                block_type = BlockType.get_block_type(value)
                if block_types[idx] < block_type:
                    block_types[idx] = block_type

    if not ret_columns:
        ret_columns = generated_default_column_names(ret_column_len)

    block_indexes = dm.append_columns(
        ret_columns, block_types
    )

    ret_blocks = [[] for _ in range(len(src_non_operable_blocks) + ret_column_len)]
    for idx, bid in enumerate(src_non_operable_blocks):
        ret_blocks[idx] = blocks[bid]

    for idx, bid in enumerate(block_indexes):
        ret_blocks[bid] = dm.blocks[bid].convert_block(apply_blocks[idx])

    return ret_blocks, dm
