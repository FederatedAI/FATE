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
import pandas as pd
import torch
from .._dataframe import DataFrame


def fillna(df: DataFrame, value, downcast=None):
    data_manager = df.data_manager
    block_indexes = data_manager.infer_operable_blocks()
    if isinstance(value, (int, float, np.int32, np.int64, np.float32, np.float64)):
        block_table = _fillna(df.block_table, value, block_indexes)
    elif isinstance(value, (list, pd.Series, dict)):
        if isinstance(value, list):
            column_names = data_manager.infer_operable_field_names()
            if len(value) != column_names:
                raise ValueError("fillna's list length should have identical column shape")
            value = dict(zip(column_names, value))
        elif isinstance(value, pd.Series):
            value = value.to_dict()
        value_indexers = dict()
        for column_name, fill_value in value.items():
            bid, offset = data_manager.loc_block(column_name)
            if bid not in value_indexers:
                value_indexers[bid] = dict()
            value_indexers[bid][offset] = fill_value

        block_table = _fillna(df.block_table, value_indexers, block_indexes)

    else:
        raise ValueError(f"Not support value type={type(value)}")

    return DataFrame(
        df._ctx,
        block_table,
        df.partition_order_mappings,
        data_manager.duplicate()
    )


def _fillna(block_table, value, block_indexes):
    block_index_set = set(block_indexes)
    if isinstance(value, (int, float, np.int32, np.int64, np.float32, np.float64)):
        def _fill(blocks):
            ret_blocks = []
            for bid, block in enumerate(blocks):
                if bid not in block_index_set:
                    ret_blocks.append(block)
                elif isinstance(block, torch.Tensor):
                    ret_blocks.append(torch.nan_to_num(block, value))
                elif isinstance(block, np.ndarray):
                    ret_blocks.append(np.nan_to_num(block, value))

            return ret_blocks

        return block_table.mapValues(_fill)
    else:
        def _fill_with_dict(blocks):
            ret_blocks = []
            for bid, block in enumerate(blocks):
                if bid not in block_index_set:
                    ret_blocks.append(block)
                elif isinstance(block, torch.Tensor):
                    block = block.clone()
                    for offset, fill_value in value.get(bid, {}).items():
                        block[:, offset] = torch.nan_to_num(block[:, offset], fill_value)
                    ret_blocks.append(block)
                elif isinstance(block, np.ndarray):
                    block = block.copy()
                    for offset, fill_value in value.get(bid, {}).items():
                        block[:, offset] = np.nan_to_num(block[:, offset], fill_value)
                    ret_blocks.append(block)

            return ret_blocks

        return block_table.mapValues(_fill_with_dict)
