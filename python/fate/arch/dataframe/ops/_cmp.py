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
import numpy as np
import pandas as pd
import torch
from .._dataframe import DataFrame
from .._dataframe import DataManager
from ..manager import BlockType
from .utils.operators import binary_operate
from .utils.series_align import series_to_ndarray
from ._compress_block import compress_blocks


def cmp_operate(lhs: DataFrame, rhs, op) -> "DataFrame":
    data_manager = lhs.data_manager
    block_indexes = data_manager.infer_operable_blocks()
    column_names = data_manager.infer_operable_field_names()

    if isinstance(rhs, (bool, int, float, np.int32, np.float32, np.int64, np.float64, np.bool_)):
        block_table = binary_operate(lhs.block_table, rhs, op, block_indexes)

    elif isinstance(rhs, (np.ndarray, list, pd.Series)):
        if isinstance(rhs, pd.Series):
            rhs = series_to_ndarray(rhs, column_names)
        if isinstance(rhs, list):
            rhs = np.array(rhs)
        if len(rhs.shape) > 2:
            raise ValueError("NdArray's Dimension should <= 2")
        if len(column_names) != rhs.size:
            raise ValueError(f"Size of List/NDArray/Series should = {len(lhs.schema.columns)}")
        rhs = rhs.reshape(-1)
        field_indexes = [data_manager.get_field_offset(name) for name in column_names]
        field_indexes_mappings = dict(zip(field_indexes, range(len(field_indexes))))
        rhs_blocks = [np.array([]) for _ in range(data_manager.block_num)]
        for bid in block_indexes:
            indexer = [field_indexes_mappings[field] for field in data_manager.get_block(bid).field_indexes]
            if BlockType.is_tensor(data_manager.get_block(bid).block_type):
                rhs_blocks[bid] = torch.Tensor(rhs[indexer])
            else:
                rhs_blocks[bid] = rhs[indexer]

        block_table = binary_operate(lhs.block_table, rhs_blocks, op, block_indexes)

    elif isinstance(rhs, DataFrame):
        other_data_manager = rhs.data_manager
        other_column_names = other_data_manager.infer_operable_field_names()
        if set(column_names) != set(other_column_names):
            raise ValueError("Comparison of two DataFrame should be identically-labeled")
        lhs_block_loc = [data_manager.loc_block(name) for name in column_names]
        rhs_block_loc = [other_data_manager.loc_block(name) for name in column_names]
        field_indexes = [data_manager.get_field_offset(name) for name in column_names]
        field_indexes_mappings = dict(zip(field_indexes, range(len(field_indexes))))
        indexers = [
            [field_indexes_mappings[field] for field in data_manager.get_block(bid).field_indexes]
            for bid in block_indexes
        ]

        block_table = _cmp_dfs(lhs.block_table, rhs.block_table, op, lhs_block_loc, rhs_block_loc,
                               block_indexes, indexers)
    else:
        raise ValueError(f"Not implement comparison of rhs type={type(rhs)}")

    block_table, data_manager = _merge_bool_blocks(block_table, data_manager, block_indexes)
    return type(lhs)(
        lhs._ctx,
        block_table,
        lhs.partition_order_mappings,
        data_manager
    )


def _merge_bool_blocks(block_table, data_manager: DataManager, block_indexes):
    """
    all blocks are bool type, they should be merge into one blocks
    """
    dst_data_manager = data_manager.duplicate()
    to_promote_types = []
    for bid in block_indexes:
        to_promote_types.append((bid, BlockType.bool))

    dst_data_manager.promote_types(to_promote_types)
    dst_block_table, dst_data_manager = compress_blocks(block_table, dst_data_manager)

    return dst_block_table, dst_data_manager


def _cmp_dfs(lhs_block_table, rhs_block_table, op,
             lhs_block_loc, rhs_block_loc,
             block_indexes, indexers):

    block_index_set = set(block_indexes)

    def _cmp_partition(l_blocks, r_blocks):
        ret_blocks = [[] for i in range(l_blocks)]
        for bid in range(len(l_blocks)):
            if bid not in block_index_set:
                ret_blocks[bid] = l_blocks[bid]

        for bid, indexer in zip(block_indexes, indexers):
            cmp_ret = torch.empty(l_blocks[bid].shape)
            for idx in indexer:
                _, l_offset = lhs_block_loc[idx]
                r_bid, r_offset = rhs_block_loc[idx]
                cmp_ret[:, l_offset] = op(l_blocks[bid][l_offset], r_blocks[r_bid][r_offset])

            ret_blocks[bid] = cmp_ret

        return ret_blocks

    block_table = lhs_block_table.join(rhs_block_table,
                                       _cmp_partition)

    return block_table
