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
from .._dataframe import DataFrame


def where(df: DataFrame, other: DataFrame):
    """
    df[mask]触发该操作
    a. mask的列可能于df不一致，这个时候，df在mask中不出现的列均为nan
        (1) columns完全对等
        (2) columns一致，但顺序不一致
        (3) mask columns数少于df columns数
    b. 当mask中某一列有false的时候，需要考虑类型问题：如果原类型为int/bool等，需要上升为float32，如果为float32，保持不变
        (1) mask 计算哪些列出现False，提前做列类型对齐
    c. 要求df与mask的key是一致的
    """
    if df.shape[0] != other.shape[0]:
        raise ValueError("Row numbers should be identical.")

    data_manager = df.data_manager
    other_data_manager = other.data_manager

    column_names = data_manager.infer_operable_field_names()
    other_column_names = other_data_manager.infer_operable_field_names()

    if column_names != other_column_names \
            and (set(column_names) == set(other_column_names)
                 or (set(column_names) & set(other_column_names)) == set(column_names)):
        other = other[column_names]
        other_column_names = column_names

    true_columns = _get_true_columns(other)
    if not true_columns:
        return df

    """
    block类型提升，此处需要得到新的block映射表
    可能的情况：类型不变，
    """

    if column_names == other_column_names:
        ...
    elif set(column_names) == set(other_column_names) \
            or (set(column_names) & set(other_column_names)) == set(column_names):
        other = other[column_names]
        ...
    else:
        ...


def _get_true_columns(df: DataFrame):
    block_table = df.block_table
    data_manager = df.data_manager
    block_index_set = set(data_manager.infer_operable_blocks())

    true_table = block_table.mapValues(
        lambda blocks: [
            block.all(axis=0) if bid in block_index_set else []
            for bid, block in enumerate(blocks)
        ]
    )

    true_values = true_table.reduce(
        lambda blocks1, blocks2:
        [
            block1 & block2 if bid in block_index_set else []
            for bid, (block1, block2) in enumerate(zip(blocks1, blocks2))
        ]
    )

    true_columns = set()
    column_names = data_manager.infer_operable_field_names()
    for name in column_names:
        _bid, _offset = data_manager.loc_block(name)
        if isinstance(true_values[_bid], torch.Tensor):
            if true_values[_bid][_offset].item():
                true_columns.add(name)

    return true_columns
