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
from fate.arch.computing import is_table
from .._dataframe import DataFrame
from ._promote_types import promote_types
from .utils.series_align import series_to_ndarray
from .utils.operators import binary_operate


def arith_operate(lhs: DataFrame, rhs, op) -> "DataFrame":
    data_manager = lhs.data_manager.duplicate()
    block_indexes = data_manager.infer_operable_blocks()
    column_names = data_manager.infer_operable_field_names()

    if isinstance(rhs, DataFrame):
        rhs_column_names = rhs.data_manager.infer_operable_field_names()
        if len(column_names) != len(rhs_column_names) or len(column_names) > 1:
            raise ValueError(f"Operation={op} of two dataframe should have same column length=1")

        rhs_block_id = rhs.data_manager.infer_operable_blocks()[0]
        block_table = _operate(lhs.block_table, rhs.block_table, op, block_indexes, rhs_block_id)
        to_promote_blocks = data_manager.try_to_promote_types(block_indexes,
                                                              rhs.data_manager.get_block(rhs_block_id).block_type)
    elif isinstance(rhs, (np.ndarray, list, pd.Series)):
        if isinstance(rhs, pd.Series):
            rhs = series_to_ndarray(rhs, column_names)
        if isinstance(rhs, list):
            rhs = np.array(rhs)
        if len(rhs.shape) > 2:
            raise ValueError("NdArray's Dimension should <= 2")
        if len(column_names) != rhs.size:
            raise ValueError(f"Size of List/NDArray should = {len(lhs.schema.columns)}")
        rhs = rhs.reshape(-1)
        field_indexes = [data_manager.get_field_offset(name) for name in column_names]
        field_indexes_mappings = dict(zip(field_indexes, range(len(field_indexes))))
        rhs_blocks = [np.array([]) for i in range(data_manager.block_num)]
        rhs_types = []
        for bid in block_indexes:
            indexer = [field_indexes_mappings[field] for field in data_manager.get_block(bid).field_indexes]
            rhs_blocks[bid] = rhs[indexer]
            rhs_types.append(rhs_blocks[bid].dtype)

        block_table = binary_operate(lhs.block_table, rhs_blocks, op, block_indexes)
        to_promote_blocks = data_manager.try_to_promote_types(block_indexes, rhs_types)

    elif isinstance(rhs, (bool, int, float, np.int32, np.float32, np.int64, np.float64, np.bool_)):
        block_table = binary_operate(lhs.block_table, rhs, op, block_indexes)
        to_promote_blocks = data_manager.try_to_promote_types(block_indexes, rhs)
    else:
        raise ValueError(f"Operation={op} between dataframe and {type(rhs)} is not implemented")

    if to_promote_blocks:
        block_table, data_manager = promote_types(block_table, data_manager, to_promote_blocks)

    return type(lhs) (
        lhs._ctx,
        block_table,
        lhs.partition_order_mappings,
        data_manager
    )


def _operate(lhs, rhs, op, block_indexes, rhs_block_id=None):
    block_index_set = set(block_indexes)
    if isinstance(rhs, list):
        op_ret = lhs.mapValues(
            lambda blocks:
            [
                op(blocks[bid], rhs[bid]) if bid in block_index_set
                                          else blocks[bid]
                for bid in range(len(blocks))
            ]
        )
    elif isinstance(rhs, (bool, int, float, np.int32, np.float32, np.int64, np.float64, np.bool_)):
        op_ret = lhs.mapValues(
            lambda blocks:
            [
                op(blocks[bid], rhs) if bid in block_index_set
                                     else blocks[bid]
                for bid in range(len(blocks))
             ]
        )
    elif is_table(rhs):
        op_ret = lhs.join(rhs,
            lambda blocks1, blocks2:
            [
                op(blocks1[bid], blocks2[rhs_block_id]) if bid in block_index_set
                                     else blocks1[bid]
                for bid in range(len(blocks1))
            ]
        )
    else:
        raise ValueError(f"Not implement type between dataframe nad {type(rhs)}")

    return op_ret
