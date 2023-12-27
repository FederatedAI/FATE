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
import torch

from collections.abc import Iterable

from .._dataframe import DataFrame
from ..manager.block_manager import Block, BlockType
from ..manager.data_manager import DataManager
from ..utils._auto_column_name_generated import generated_default_column_names


def apply_row(
    df: "DataFrame", func, columns: list = None, with_label=False, with_weight=False, enable_type_align_checking=True
) -> "DataFrame":
    """
    In current version, assume that the apply_row results' lengths are equal
    """
    data_manager = df.data_manager
    dst_data_manager, _ = data_manager.derive_new_data_manager(
        with_sample_id=True, with_match_id=True, with_label=not with_label, with_weight=not with_weight, columns=None
    )

    non_operable_field_names = dst_data_manager.get_field_name_list()
    non_operable_blocks = [
        data_manager.loc_block(field_name, with_offset=False) for field_name in non_operable_field_names
    ]
    fields_loc = data_manager.get_fields_loc(
        with_sample_id=False, with_match_id=False, with_label=with_label, with_weight=with_weight
    )

    fields_name = data_manager.get_field_name_list(
        with_sample_id=False, with_match_id=False, with_label=with_label, with_weight=with_weight
    )

    operable_blocks = sorted(list(set(data_manager.loc_block(fields_name, with_offset=False))))
    is_numeric = True
    for bid in operable_blocks:
        if not data_manager.get_block(bid).is_numeric():
            is_numeric = False
            break
    block_column_in_orders = list()
    if is_numeric:
        for bid in operable_blocks:
            field_indexes = data_manager.get_block(bid).field_indexes
            block_column_in_orders.extend([data_manager.get_field_name(field_index) for field_index in field_indexes])

    _apply_func = functools.partial(
        _apply,
        func=func,
        src_operable_blocks=operable_blocks,
        src_field_names=fields_name,
        src_fields_loc=fields_loc,
        src_non_operable_blocks=non_operable_blocks,
        ret_columns=columns,
        dst_dm=dst_data_manager,
        is_numeric=is_numeric,
        need_shuffle=True if block_column_in_orders == fields_name else False,
        block_column_in_orders=block_column_in_orders,
        enable_type_align_checking=enable_type_align_checking,
    )

    dst_block_table_with_dm = df.block_table.mapValues(_apply_func)

    dst_data_manager = dst_block_table_with_dm.first()[1][1]
    dst_block_table = dst_block_table_with_dm.mapValues(lambda blocks_with_dm: blocks_with_dm[0])

    return DataFrame(df._ctx, dst_block_table, df.partition_order_mappings, dst_data_manager)


def _apply(
    blocks,
    func=None,
    src_operable_blocks=None,
    src_field_names=None,
    src_fields_loc=None,
    src_non_operable_blocks=None,
    ret_columns=None,
    dst_dm: "DataManager" = None,
    is_numeric=True,
    block_column_in_orders=None,
    need_shuffle=False,
    enable_type_align_checking=True,
):
    dm = dst_dm.duplicate()
    apply_blocks = []

    if is_numeric:
        apply_data = []
        for bid in src_operable_blocks:
            apply_data.append(blocks[bid])
        apply_data = torch.hstack(apply_data)
        apply_data = pd.DataFrame(apply_data, columns=block_column_in_orders)
        if need_shuffle:
            apply_data = apply_data[src_field_names]
    else:
        lines = len(blocks[0])
        flat_blocks = [Block.transform_block_to_list(block) for block in blocks]
        apply_data = [[] for _ in range(lines)]
        for bid, offset in src_fields_loc:
            for lid in range(lines):
                apply_data[lid].append(flat_blocks[bid][lid][offset])

        apply_data = pd.DataFrame(apply_data, columns=src_field_names)

    apply_ret = apply_data.apply(lambda row: func(row), axis=1).values.tolist()

    if isinstance(apply_ret[0], Iterable):
        first_row = list(apply_ret[0])
        ret_column_len = len(first_row)
        block_types = [
            BlockType.np_object if BlockType.is_arr(value) else BlockType.get_block_type(value) for value in first_row
        ]
        apply_blocks = [[] for _ in range(ret_column_len)]
        for ret in apply_ret:
            for idx, value in enumerate(ret):
                apply_blocks[idx].append([value])

                if enable_type_align_checking:
                    block_type = BlockType.np_object if BlockType.is_arr(value) else BlockType.get_block_type(value)
                    if block_types[idx] < block_type:
                        block_types[idx] = block_type
    else:
        block_types = [
            BlockType.np_object if BlockType.is_arr(apply_ret[0]) else BlockType.get_block_type(apply_ret[0])
        ]
        apply_blocks.append([[ret] for ret in apply_ret])
        ret_column_len = 1

        if enable_type_align_checking:
            for ret in apply_ret:
                block_type = BlockType.np_object if BlockType.is_arr(ret) else BlockType.get_block_type(ret)
                if block_types[0] < block_type:
                    block_types[0] = block_type

    if not ret_columns:
        ret_columns = generated_default_column_names(ret_column_len)

    block_indexes = dm.append_columns(ret_columns, block_types)

    ret_blocks = [[] for _ in range(len(src_non_operable_blocks) + ret_column_len)]
    for idx, bid in enumerate(src_non_operable_blocks):
        ret_blocks[idx] = blocks[bid]

    for idx, bid in enumerate(block_indexes):
        if dm.blocks[bid].is_phe_tensor():
            single_value = apply_blocks[idx][0][0]
            dm.blocks[bid].set_extra_kwargs(
                pk=single_value.pk,
                evaluator=single_value.evaluator,
                coder=single_value.coder,
                dtype=single_value.dtype,
                device=single_value.device,
            )
            ret = [v[0]._data for v in apply_blocks[idx]]
            ret_blocks[bid] = dm.blocks[bid].convert_block(ret)
            # ret_blocks[bid] = dm.blocks[bid].convert_to_phe_tensor(ret, shape=(len(ret), 1))
        else:
            ret_blocks[bid] = dm.blocks[bid].convert_block(apply_blocks[idx])

    return ret_blocks, dm
