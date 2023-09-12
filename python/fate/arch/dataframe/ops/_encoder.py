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

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from typing import Union
from ._compress_block import compress_blocks
from .._dataframe import DataFrame
from ..manager import BlockType, DataManager


BUCKETIZE_RESULT_TYPE = "int32"


def get_dummies(df: "DataFrame", dtype="int32"):
    data_manager = df.data_manager
    block_indexes = data_manager.infer_operable_blocks()
    field_names = data_manager.infer_operable_field_names()

    if len(field_names) != 1:
        raise ValueError(f"get_dummies only support single column, but {len(field_names)} columns are found.")

    categories = _get_categories(df.block_table, block_indexes)[0][0]
    dst_field_names = ["_".join(map(str, [field_names[0], c])) for c in categories]
    dst_data_manager = data_manager.duplicate()
    dst_data_manager.pop_blocks(block_indexes)
    dst_data_manager.append_columns(dst_field_names, block_types=BlockType.get_block_type(dtype))

    block_table = _one_hot_encode(df.block_table, block_indexes, dst_data_manager, [[categories]], dtype=dtype)

    return DataFrame(
        df._ctx,
        block_table,
        partition_order_mappings=df.partition_order_mappings,
        data_manager=dst_data_manager
    )


def _get_categories(block_table, block_indexes):
    block_index_set = set(block_indexes)

    def _mapper(blocks):
        categories_ = []
        for bid, block in enumerate(blocks):
            if bid not in block_index_set:
                continue

            enc = OneHotEncoder()
            cate_block = enc.fit(block).categories_
            categories_.append([set(cate) for cate in cate_block])

        return categories_

    def _reducer(categories1_, categories2_):
        categories_ = []
        for cate_block1, cate_block2 in zip(categories1_, categories2_):
            cate_block = [cate1 | cate2 for cate1, cate2 in zip(cate_block1, cate_block2)]
            categories_.append(cate_block)

        return categories_

    categories = block_table.mapValues(_mapper).reduce(_reducer)

    categories = [[sorted(cate) for cate in cate_block] for cate_block in categories]

    return categories


def _one_hot_encode(block_table, block_indexes, data_manager, categories, dtype):
    categories = [np.array(category) for category in categories]
    block_index_set = set(block_indexes)

    def _encode(blocks):
        ret_blocks = []
        enc_blocks = []
        idx = 0
        for bid, block in enumerate(blocks):
            if bid not in block_index_set:
                ret_blocks.append(block)
                continue

            enc = OneHotEncoder(dtype=dtype)
            enc.fit([[1]])  # one hot encoder need to fit first.
            enc.categories_ = categories[idx]
            idx += 1
            enc_blocks.append(enc.transform(block).toarray())

        ret_blocks.append(data_manager.blocks[-1].convert_block(np.hstack(enc_blocks)))

        return ret_blocks

    return block_table.mapValues(_encode)


def bucketize(df: DataFrame, boundaries: Union[pd.DataFrame, dict]):
    if isinstance(boundaries, pd.DataFrame):
        boundaries = dict([(_name, boundaries[_name].tolist()) for _name in boundaries])
    elif not isinstance(boundaries, dict):
        raise ValueError("boundaries should be pd.DataFrame or dict")

    data_manager = df.data_manager.duplicate()
    field_names = list(filter(lambda field_name: field_name in boundaries, data_manager.infer_operable_field_names()))
    blocks_loc = data_manager.loc_block(field_names)

    _boundaries_list = []
    for name, (_bid, _) in zip(field_names, blocks_loc):
        if BlockType.is_tensor(data_manager.blocks[_bid].block_type):
            _boundary = torch.tensor(boundaries[name])
            _boundary[-1] = torch.inf
        else:
            _boundary = np.array(boundaries[name])
            _boundary[-1] = np.inf

        _boundaries_list.append((_bid, _, _boundary))

    narrow_blocks, dst_blocks = data_manager.split_columns(
        field_names, BlockType.get_block_type(BUCKETIZE_RESULT_TYPE)
    )

    def _mapper(
        blocks, boundaries_list: list = None, narrow_loc: list = None, dst_bids: list = None, dm: DataManager = None
    ):
        ret_blocks = []
        for block in blocks:
            if isinstance(block, torch.Tensor):
                ret_blocks.append(block.clone())
            elif isinstance(block, np.ndarray):
                ret_blocks.append(block.copy())
            else:
                ret_blocks.append(block)

        for i in range(len(ret_blocks), dm.block_num):
            ret_blocks.append([])

        for bid, offsets in narrow_loc:
            ret_blocks[bid] = ret_blocks[bid][:, offsets]

        for dst_bid, (src_bid, src_offset, boundary) in zip(dst_bids, boundaries_list):
            if isinstance(blocks[src_bid], torch.Tensor):
                ret = torch.bucketize(blocks[src_bid][:, [src_offset]], boundary, out_int32=False)
            else:
                ret = torch.bucketize(blocks[src_bid][:, [src_offset]], boundary)

            ret_blocks[dst_bid] = dm.blocks[dst_bid].convert_block(ret)

        return ret_blocks

    bucketize_mapper = functools.partial(
        _mapper, boundaries_list=_boundaries_list, narrow_loc=narrow_blocks, dst_bids=dst_blocks, dm=data_manager
    )

    block_table = df.block_table.mapValues(bucketize_mapper)

    block_indexes = data_manager.infer_operable_blocks()
    if len(block_indexes) > 1:
        to_promote_types = []
        for bid in block_indexes:
            to_promote_types.append((bid, BlockType.get_block_type(BUCKETIZE_RESULT_TYPE)))

        data_manager.promote_types(to_promote_types)
        block_table, data_manager = compress_blocks(block_table, data_manager)

    return DataFrame(
        df._ctx, block_table, partition_order_mappings=df.partition_order_mappings, data_manager=data_manager
    )


def bucketize(df: DataFrame, boundaries: Union[pd.DataFrame, dict]):
    if isinstance(boundaries, pd.DataFrame):
        boundaries = dict([(_name, boundaries[_name].tolist()) for _name in boundaries])
    elif not isinstance(boundaries, dict):
        raise ValueError("boundaries should be pd.DataFrame or dict")

    data_manager = df.data_manager.duplicate()
    field_names = list(filter(lambda field_name: field_name in boundaries, data_manager.infer_operable_field_names()))
    blocks_loc = data_manager.loc_block(field_names)

    _boundaries_list = []
    for name, (_bid, _) in zip(field_names, blocks_loc):
        if BlockType.is_tensor(data_manager.blocks[_bid].block_type):
            _boundary = torch.tensor(boundaries[name], dtype=torch.float64)
            _boundary[-1] = torch.inf
        else:
            _boundary = np.array(boundaries[name], dtype=np.float64)
            _boundary[-1] = np.inf

        _boundaries_list.append((_bid, _, _boundary))

    narrow_blocks, dst_blocks = data_manager.split_columns(
        field_names, BlockType.get_block_type(BUCKETIZE_RESULT_TYPE)
    )

    def _mapper(
        blocks, boundaries_list: list = None, narrow_loc: list = None, dst_bids: list = None, dm: DataManager = None
    ):
        ret_blocks = [block for block in blocks]

        for i in range(len(ret_blocks), dm.block_num):
            ret_blocks.append([])

        for bid, offsets in narrow_loc:
            ret_blocks[bid] = ret_blocks[bid][:, offsets]

        for dst_bid, (src_bid, src_offset, boundary) in zip(dst_bids, boundaries_list):
            if isinstance(blocks[src_bid], torch.Tensor):
                ret = torch.bucketize(blocks[src_bid][:, [src_offset]], boundary, out_int32=False)
            else:
                ret = np.digitize(blocks[src_bid][:, [src_offset]], boundary)

            ret_blocks[dst_bid] = dm.blocks[dst_bid].convert_block(ret)

        return ret_blocks

    bucketize_mapper = functools.partial(
        _mapper, boundaries_list=_boundaries_list, narrow_loc=narrow_blocks, dst_bids=dst_blocks, dm=data_manager
    )

    block_table = df.block_table.mapValues(bucketize_mapper)

    block_indexes = data_manager.infer_operable_blocks()
    if len(block_indexes) > 1:
        to_promote_types = []
        for _bid in block_indexes:
            to_promote_types.append((_bid, data_manager.get_block(_bid).block_type))

        data_manager.promote_types(to_promote_types)
        block_table, data_manager = compress_blocks(block_table, data_manager)

    return DataFrame(
        df._ctx, block_table, partition_order_mappings=df.partition_order_mappings, data_manager=data_manager
    )
