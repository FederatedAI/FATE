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

import numpy as np
import pandas as pd
import torch
from .._dataframe import DataFrame
from ..manager import DataManager


FLOATING_POINT_ZERO = 1e-14


def min(df: "DataFrame") -> "pd.Series":
    data_manager = df.data_manager
    operable_blocks = data_manager.infer_operable_blocks()

    def _mapper(blocks, op_bids):
        ret = []
        for bid in op_bids:
            if isinstance(blocks[bid], torch.Tensor):
                ret.append(blocks[bid].min(axis=0).values)
            else:
                ret.append(blocks[bid].min(axis=0))

        return ret

    def _reducer(blocks1, blocks2):
        ret = []
        for block1, block2 in zip(blocks1, blocks2):
            if isinstance(block1, torch.Tensor):
                ret.append(torch.minimum(block1, block2))
            else:
                ret.append(np.minimum(block1, block2))

        return ret

    mapper_func = functools.partial(_mapper, op_bids=operable_blocks)

    reduce_ret = df.block_table.mapValues(mapper_func).reduce(_reducer)
    return _post_process(reduce_ret, operable_blocks, data_manager)


def max(df: "DataFrame") -> "pd.Series":
    data_manager = df.data_manager
    operable_blocks = data_manager.infer_operable_blocks()

    def _mapper(blocks, op_bids):
        ret = []
        for bid in op_bids:
            if isinstance(blocks[bid], torch.Tensor):
                ret.append(blocks[bid].max(axis=0).values)
            else:
                ret.append(blocks[bid].max(axis=0))

        return ret

    def _reducer(blocks1, blocks2):
        ret = []
        for block1, block2 in zip(blocks1, blocks2):
            if isinstance(block1, torch.Tensor):
                ret.append(torch.maximum(block1, block2))
            else:
                ret.append(np.maximum(block1, block2))

        return ret

    mapper_func = functools.partial(_mapper, op_bids=operable_blocks)

    reduce_ret = df.block_table.mapValues(mapper_func).reduce(_reducer)
    return _post_process(reduce_ret, operable_blocks, data_manager)


def sum(df: DataFrame) -> "pd.Series":
    data_manager = df.data_manager
    operable_blocks = data_manager.infer_operable_blocks()

    def _mapper(blocks, op_bids):
        ret = []
        for bid in op_bids:
            ret.append(blocks[bid].sum(axis=0))

        return ret

    def _reducer(blocks1, blocks2):
        return [block1 + block2 for block1, block2 in zip(blocks1, blocks2)]

    mapper_func = functools.partial(_mapper, op_bids=operable_blocks)

    reduce_ret = df.block_table.mapValues(mapper_func).reduce(_reducer)
    return _post_process(reduce_ret, operable_blocks, data_manager)


def mean(df: "DataFrame") -> "pd.Series":
    return sum(df) / df.shape[0]


def var(df: "DataFrame", ddof=1) -> "pd.Series":
    data_manager = df.data_manager
    operable_blocks = data_manager.infer_operable_blocks()
    n = df.shape[0]

    def _mapper(blocks, op_bids):
        ret = []
        for bid in op_bids:
            block = blocks[bid]
            if isinstance(block, torch.Tensor):
                ret.append(
                    (
                        torch.sum(torch.square(block), dim=0, keepdim=True),
                        torch.sum(block, dim=0, keepdim=True),
                    )
                )
            else:
                ret.append((np.sum(np.square(block), axis=0), np.sum(block, axis=0)))

        return ret

    def _reducer(blocks1, block2):
        ret = []
        for block1, block2 in zip(blocks1, block2):
            if isinstance(block1, torch.Tensor):
                ret.append((torch.add(block1[0], block2[0]), torch.add(block1[1], block2[1])))
            else:
                ret.append((np.add(block1[0], block2[0]), np.add(block1[1], block2[1])))

        return ret

    mapper_func = functools.partial(_mapper, op_bids=operable_blocks)
    reduce_ret = df.block_table.mapValues(mapper_func).reduce(_reducer)

    ret_blocks = []
    for lhs, rhs in reduce_ret:
        if isinstance(lhs, torch.Tensor):
            rhs = torch.mul(torch.square(torch.div(rhs, n)), n)
            ret_blocks.append(torch.div(torch.sub(lhs, rhs), n - ddof))
        else:
            rhs = np.mul(np.square(np.div(rhs, n)), n)
            ret_blocks.append(np.div(np.sub(lhs, rhs), n - ddof))

    return _post_process(ret_blocks, operable_blocks, data_manager)


def std(df: "DataFrame", ddof=1) -> "pd.Series":
    return var(df, ddof=ddof) ** 0.5


def skew(df: "DataFrame", unbiased=False):
    data_manager = df.data_manager
    n = df.shape[0]

    if unbiased and n < 3:
        field_names = data_manager.infer_operable_field_names()
        return pd.Series([np.nan for _ in range(len(field_names))], index=field_names)

    _mean = mean(df)
    m1 = df - _mean
    m2 = (m1**2).mean()
    m3 = (m1**3).mean()

    """
    if abs(value) in m2 < eps=1e-14, we regard it as 0, but eps=1e-14 should be global instead of this file.
    """
    non_zero_mask = abs(m2) >= FLOATING_POINT_ZERO
    m3[~non_zero_mask] = 0
    m2[~non_zero_mask] = 1

    if unbiased:
        return (n * (n - 1)) ** 0.5 / (n - 2) * (m3 / m2**1.5)
    else:
        return m3 / m2**1.5


def kurt(df: "DataFrame", unbiased=False):
    data_manager = df.data_manager
    n = df.shape[0]
    if unbiased and n < 4:
        field_names = data_manager.infer_operable_field_names()
        return pd.Series([np.nan for _ in range(len(field_names))], index=field_names)

    _mean = mean(df)
    m1 = df - _mean
    m2 = m1**2
    m4 = m2**2
    m2 = m2.mean()
    m4 = m4.mean()

    non_zero_mask = abs(m2) >= FLOATING_POINT_ZERO
    m4[~non_zero_mask] = 0
    m2[~non_zero_mask] = 1

    if unbiased:
        return (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * m4 / m2**2 - 3 * (n - 1))
    else:
        return m4 / m2**4 - 3


def variation(df: "DataFrame", ddof=1):
    return std(df, ddof=ddof) / mean(df)


def describe(df: "DataFrame", ddof=1, unbiased=False):
    stat_metrics = dict()
    stat_metrics["sum"] = sum(df)
    stat_metrics["min"] = min(df)
    stat_metrics["max"] = max(df)
    stat_metrics["mean"] = mean(df)
    stat_metrics["std"] = std(df, ddof=ddof)
    stat_metrics["var"] = var(df, ddof=ddof)
    stat_metrics["variation"] = variation(df, ddof=ddof)
    stat_metrics["skew"] = skew(df, unbiased=unbiased)
    stat_metrics["kurt"] = kurt(df, unbiased=unbiased)
    stat_metrics["na_count"] = df.isna().sum()

    return pd.DataFrame(stat_metrics)


def _post_process(reduce_ret, operable_blocks, data_manager: "DataManager") -> "pd.Series":
    field_names = data_manager.infer_operable_field_names()
    field_indexes = [data_manager.get_field_offset(name) for name in field_names]
    field_indexes_loc = dict(zip(field_indexes, range(len(field_indexes))))
    ret = [[] for i in range(len(field_indexes))]

    reduce_ret = [r.reshape(-1).tolist() for r in reduce_ret]
    for idx, bid in enumerate(operable_blocks):
        field_indexes = data_manager.blocks[bid].field_indexes
        for offset, field_index in enumerate(field_indexes):
            loc = field_indexes_loc[field_index]
            ret[loc] = reduce_ret[idx][offset]

    return pd.Series(ret, index=field_names)
