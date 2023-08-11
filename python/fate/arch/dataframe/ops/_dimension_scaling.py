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
import copy
import functools
from typing import List
import pandas as pd
import torch
from sklearn.utils import resample
from .._dataframe import DataFrame
from ..manager.data_manager import DataManager
from ._compress_block import compress_blocks
from ._indexer import get_partition_order_by_raw_table
from ._promote_types import promote_partial_block_types
from ._set_item import set_item
from fate.arch.tensor import DTensor


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
    frame_0 = data_frames[0]
    data_frames = list(filter(lambda df: df.shape[0], data_frames))
    if len(data_frames) <= 1:
        return frame_0 if not data_frames else data_frames[0]

    def _align_blocks(blocks, align_fields_loc=None, full_block_migrate_set=None, dst_dm: DataManager = None):
        ret_blocks, lines = [], None
        for dst_bid, block in enumerate(dst_dm.blocks):
            _field_indexes = block.field_indexes
            _src_bid = align_fields_loc[_field_indexes[0]][0]
            if _src_bid in full_block_migrate_set:
                ret_blocks.append(blocks[_src_bid])
            else:
                _align_block = []
                lines = len(blocks[0]) if lines is None else lines
                for lid in range(lines):
                    row = []
                    for _field_index in _field_indexes:
                        _src_bid, _offset = align_fields_loc[_field_index]
                        row.append(blocks[_src_bid][lid][_offset].item() if isinstance(blocks[_src_bid], torch.Tensor)
                                   else blocks[_src_bid][lid][_offset])

                    _align_block.append(row)

                ret_blocks.append(dst_dm.blocks[dst_bid].convert_block(_align_block))

        return ret_blocks

    l_df = data_frames[0]
    data_manager = l_df.data_manager.duplicate()
    l_fields_loc = data_manager.get_fields_loc()
    l_field_names = data_manager.get_field_name_list()
    l_field_types = [data_manager.get_block(_bid).block_type for _bid, _ in l_fields_loc]
    l_block_table = l_df.block_table
    type_change = False
    for r_df in data_frames[1:]:
        if set(l_df.schema.columns) != set(r_df.schema.columns):
            raise ValueError("vstack of dataframes should have same schemas")

        for idx, field_name in enumerate(l_field_names):
            block_type = r_df.data_manager.get_block(
                r_df.data_manager.loc_block(field_name, with_offset=False)).block_type
            if block_type > l_field_types[idx]:
                l_field_types[idx] = block_type
                type_change = True

    if type_change:
        changed_fields, changed_block_types, changed_fields_loc = [], [], []
        changed_block_types = []
        for idx in range(len(l_field_names)):
            field_name, block_type, (bid, offset) = l_field_names[idx], l_field_types[idx], l_fields_loc[idx]
            if block_type != data_manager.get_block(bid).block_type:
                changed_fields.append(field_name)
                changed_block_types.append(block_type)
                changed_fields_loc.append((bid, offset))

        narrow_blocks, dst_blocks = data_manager.split_columns(changed_fields, changed_block_types)
        l_block_table = promote_partial_block_types(l_block_table, narrow_blocks=narrow_blocks, dst_blocks=dst_blocks,
                                                    data_manager=data_manager, dst_fields_loc=changed_fields_loc)

    l_flatten_func = functools.partial(_flatten_partition, block_num=data_manager.block_num)
    l_flatten = l_block_table.mapPartitions(l_flatten_func, use_previous_behavior=False)

    for r_df in data_frames[1:]:
        r_field_names = r_df.data_manager.get_field_name_list()
        r_fields_loc = r_df.data_manager.get_fields_loc()
        r_field_types = [data_manager.get_block(_bid).block_type for _bid, _ in r_fields_loc]
        r_type_change = False if l_field_types != r_field_types else True
        r_block_table = r_df.block_table
        if l_field_names != r_field_names or r_type_change:
            shuffle_r_fields_loc, full_migrate_set = [() for _ in range(len(r_field_names))], set()
            for field_name, loc in zip(r_field_names, r_fields_loc):
                l_offset = data_manager.get_field_offset(field_name)
                shuffle_r_fields_loc[l_offset] = loc

            for bid in range(r_df.data_manager.block_num):
                r_field_indexes = r_df.data_manager.get_block(bid).field_indexes
                field_indexes = [data_manager.get_field_offset(r_field_names[idx]) for idx in r_field_indexes]
                l_bid = data_manager.loc_block(r_field_names[r_field_indexes[0]], with_offset=False)
                if field_indexes == data_manager.get_block(l_bid).field_indexes:
                    full_migrate_set.add(bid)

            _align_func = functools.partial(_align_blocks, align_fields_loc=shuffle_r_fields_loc,
                                            full_block_migrate_set=full_migrate_set, dst_dm=data_manager)
            r_block_table = r_block_table.mapValues(_align_func)

        r_flatten_func = functools.partial(_flatten_partition, block_num=data_manager.block_num)
        r_flatten = r_block_table.mapPartitions(r_flatten_func, use_previous_behavior=False)
        l_flatten = l_flatten.union(r_flatten)

    partition_order_mappings = get_partition_order_by_raw_table(l_flatten, data_manager.block_row_size)
    _convert_to_block_func = functools.partial(to_blocks, dm=data_manager, partition_mappings=partition_order_mappings)
    block_table = l_flatten.mapPartitions(_convert_to_block_func, use_previous_behavior=False)
    block_table, data_manager = compress_blocks(block_table, data_manager)

    return DataFrame(
        l_df._ctx,
        block_table,
        partition_order_mappings,
        data_manager
    )


def drop(df: "DataFrame", index: "DataFrame" = None) -> "DataFrame":
    if index.shape[0] == 0:
        return DataFrame(
            df._ctx,
            block_table=df.block_table,
            partition_order_mappings=copy.deepcopy(df.partition_order_mappings),
            data_manager=df.data_manager.duplicate()
        )

    if index.shape[0] == df.shape[0]:
        return df.empty_frame()

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
    partition_order_mappings = get_partition_order_by_raw_table(
        drop_flatten, data_manager.block_row_size
    ) if drop_flatten.count() else dict()

    _convert_to_block_func = functools.partial(to_blocks,
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


def retrieval_row(df: "DataFrame", indexer: "DTensor"):
    if indexer.shape[1] != 1:
        raise ValueError("Row indexing by DTensor should have only one column filling with True/False")

    def _retrieval(blocks, t: torch.Tensor):
        index = t.reshape(-1).tolist()
        ret_blocks = [block[index] for block in blocks]

        return ret_blocks

    _retrieval_func = functools.partial(_retrieval)
    retrieval_block_table = df.block_table.join(indexer.shardings._data, _retrieval_func)

    _flatten_func = functools.partial(_flatten_partition, block_num=df.data_manager.block_num)
    retrieval_raw_table = retrieval_block_table.mapPartitions(_flatten_func, use_previous_behavior=False)

    if retrieval_raw_table.count() == 0:
        return df.empty_frame()

    partition_order_mappings = get_partition_order_by_raw_table(retrieval_raw_table)
    to_blocks_func = functools.partial(to_blocks, dm=df.data_manager, partition_mappings=partition_order_mappings)

    block_table = retrieval_raw_table.mapPartitions(to_blocks_func, use_previous_behavior=False)

    return DataFrame(
        df._ctx,
        block_table,
        partition_order_mappings,
        df.data_manager
    )


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


def to_blocks(kvs, dm: DataManager = None, partition_mappings: dict = None):
    ret_blocks = [[] for _ in range(dm.block_num)]

    block_id = None
    for lid, (sample_id, value) in enumerate(kvs):
        if block_id is None:
            block_id = partition_mappings[sample_id]["start_block_id"]
        ret_blocks[0].append(sample_id)
        for bid, buf in enumerate(value):
            ret_blocks[bid + 1].append(buf)

        if (lid + 1) % dm.block_row_size == 0:
            yield block_id, dm.convert_to_blocks(ret_blocks)
            ret_blocks = [[] for i in range(dm.block_num)]
            block_id += 1

    if ret_blocks[0]:
        yield block_id, dm.convert_to_blocks(ret_blocks)
