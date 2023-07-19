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
from typing import List
import pandas as pd
import torch
from sklearn.utils import resample
from .._dataframe import DataFrame
from ..manager.data_manager import DataManager
from ._compress_block import compress_blocks
from ._indexer import get_partition_order_by_raw_table
from ._set_item import set_item


def hstack(data_frames: List["DataFrame"]) -> "DataFrame":
    if len(data_frames) == 1:
        return data_frames[0]

    l_df = DataFrame(data_frames[0]._ctx,
                     data_frames[0].block_table,
                     data_frames[0].partition_order_mappings,
                     data_frames[0].data_manager.duplicate())

    column_set = set(l_df.schema.columns)
    for r_df in data_frames[1:]:
        other_column_set = set(r_df.schema.columns)
        if column_set & other_column_set:
            raise ValueError("Hstack does not support duplicate columns")

        set_item(l_df, r_df.schema.columns.tolist(), r_df, 1)

        column_set |= other_column_set

    data_manager = l_df.data_manager
    block_table = l_df.block_table

    block_table, data_manager = compress_blocks(block_table, data_manager)

    l_df.block_table = block_table
    l_df.data_manager = data_manager

    return l_df


def vstack(data_frames: List["DataFrame"]) -> "DataFrame":
    if len(data_frames[0]) == 1:
        return data_frames[0]

    def _align_blocks(blocks, src_fields_loc=None, src_dm: DataManager=None, dst_dm: DataManager=None):
        ret_blocks = []
        lines = None
        for dst_bid, block in enumerate(dst_dm.blocks):
            field_indexes = block.field_indexes
            src_bid = src_fields_loc[field_indexes[0]][0]
            if src_dm.blocks[src_bid].field_indexes == field_indexes:
                ret_blocks.append(blocks[src_bid])
            else:
                block_buf = []
                lines = len(blocks[0]) if lines is None else lines

                for lid in range(lines):
                    row = []
                    for field_index in field_indexes:
                        src_bid, offset = src_fields_loc[field_index]
                        if isinstance(blocks[src_bid], torch.Tensor):
                            row.append(blocks[src_bid][lid][offset].item())
                        else:
                            row.append(blocks[src_bid][lid][offset])

                    block_buf.append(row)

                ret_blocks.append(dst_dm.blocks[dst_bid].convert_block(block_buf))

        return ret_blocks

    l_df = data_frames[0]
    data_manager = l_df.data_manager
    l_fields_loc = data_manager.get_fields_loc()

    l_flatten_func = functools.partial(_flatten_partition, block_num=data_manager.block_num)
    l_flatten = l_df.block_table.mapPartitions(l_flatten_func, use_previous_behavior=False)

    for r_df in data_frames[1:]:
        if l_df.schema != r_df.schema:
            raise ValueError("Vstack of two dataframe with different schemas")

        r_fields_loc = r_df.data_manager.get_fields_loc()
        block_table = r_df.block_table
        if l_fields_loc != r_fields_loc:
            _align_func = functools.partial(_align_blocks, src_fields_loc=r_fields_loc, dm=data_manager)
            block_table = block_table.mapValues(_align_func)

        r_flatten_func = functools.partial(_flatten_partition, block_num=data_manager.block_num)
        r_flatten = block_table.mapPartitions(r_flatten_func, use_previous_behavior=False)
        l_flatten = l_flatten.union(r_flatten)

        # TODO: data-manager support align blocks first
        # TODO: a fast way of vstack is just increase partition_id in r_df, then union,
        #  but data in every partition may be unbalance, so we use a more slow way by flatten data first

    partition_order_mappings = get_partition_order_by_raw_table(l_flatten)
    _convert_to_block_func = functools.partial(_to_blocks,
                                               dm=data_manager,
                                               partition_mappings=partition_order_mappings)

    block_table = l_flatten.mapPartitions(_convert_to_block_func,
                                          use_previous_behavior=False)
    return DataFrame(
        l_df._ctx,
        block_table,
        partition_order_mappings,
        data_manager
    )


def drop(df: "DataFrame", index: "DataFrame" = None) -> "DataFrame":
    data_manager = df.data_manager.duplicate()
    l_flatten_func = functools.partial(
        _flatten_partition,
        block_num=data_manager.block_num
    )
    l_flatten_table = df.block_table.mapPartitions(l_flatten_func, use_previous_behavior=False)

    r_flatten_func = functools.partial(
        _flatten_partition,
        block_num=index.data_manager.block_num
    )
    r_flatten_table = index.block_table.mapPartitions(r_flatten_func, use_previous_behavior=False)

    drop_flatten = l_flatten_table.subtractByKey(r_flatten_table)
    partition_order_mappings = get_partition_order_by_raw_table(drop_flatten) if drop_flatten.count() else dict()

    _convert_to_block_func = functools.partial(_to_blocks,
                                               dm=data_manager,
                                               partition_mappings=partition_order_mappings)

    block_table = drop_flatten.mapPartitions(_convert_to_block_func,
                                             use_previous_behavior=False)

    return DataFrame(
        df._ctx,
        block_table,
        partition_order_mappings,
        data_manager
    )


def sample(df: "DataFrame", n=None, frac: float =None, random_state=None) -> "DataFrame":
    """
    only support down sample, n should <= df.shape, or fact = 1
    """

    if n is not None and frac is not None:
        raise ValueError("sample's parameters n and frac should not be set in the same time.")

    if frac is not None:
        if frac > 1:
            raise ValueError(f"sample's parameter frac={frac} should <= 1.0")
        n = max(1, int(df.shape[0] * frac))

    if n > df.shape[0]:
        raise ValueError(f"sample's parameter n={n} > data size={df.shape[0]}")

    if n == 0:
        raise ValueError(f"sample's parameter n={n} should >= 1")

    indexer = list(df.get_indexer(target="sample_id").collect())
    sample_indexer = resample(indexer, replace=False, n_samples=n, random_state=random_state)

    sample_indexer = df._ctx.computing.parallelize(sample_indexer,
                                                   include_key=True,
                                                   partition=df.block_table.partitions)

    sample_frame = df.loc(sample_indexer)

    return sample_frame


def _flatten_partition(kvs, block_num=0):
    _flattens = []
    for partition_id, blocks in kvs:
        lines = len(blocks[0])
        for idx in range(lines):
            sample_id = blocks[0][idx]
            row = []
            for bid in range(1, block_num):
                if isinstance(blocks[bid], pd.Index):
                    row.append(blocks[bid][idx])
                else:
                    row.append(blocks[bid][idx].tolist())

            _flattens.append((sample_id, row))

    return _flattens


def _to_blocks(kvs, dm: DataManager=None, partition_mappings: dict=None):
    ret_blocks = [[] for i in range(dm.block_num)]

    partition_id = None
    for sample_id, value in kvs:
        if partition_id is None:
            partition_id = partition_mappings[sample_id]["block_id"]
        ret_blocks[0].append(sample_id)
        for bid, buf in enumerate(value):
            ret_blocks[bid + 1].append(buf)

    ret_blocks = dm.convert_to_blocks(ret_blocks)

    return [(partition_id, ret_blocks)]
