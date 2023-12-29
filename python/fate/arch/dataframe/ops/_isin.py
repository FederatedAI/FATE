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
from ._compress_block import compress_blocks
from .._dataframe import DataFrame


def isin(df: DataFrame, values):
    """
    support types are: scalar、list、series、dict
    note: torch.isin and np.isin does not support nan, so use torch.isnan and np.isnan.
          value type of set and list are not same, e.g.: {1.0}/[1.0] may lead to different result,
              so does ont change to set now
    """
    if isinstance(values, (list, dict, pd.Series)):
        data_manager = df.data_manager
        block_indexes = data_manager.infer_operable_blocks()
        if isinstance(values, pd.Series):
            values = values.to_dict()
        if isinstance(values, dict):
            value_indexers = dict()
            for column_name, in_value in values.items():
                bid, offset = data_manager.loc_block(column_name)
                if bid not in value_indexers:
                    value_indexers[bid] = dict()
                value_indexers[bid][offset] = in_value
            block_table = _isin(df.block_table, value_indexers, block_indexes)
        else:
            block_table = _isin(df.block_table, values, block_indexes)
    else:
        raise ValueError(f"isin only support type in [list, dict, pandas.Series], but {type(values)} was found")

    dst_data_manager = data_manager.duplicate()
    to_promote_types = []
    for bid in block_indexes:
        to_promote_types.append((bid, torch.bool))

    dst_data_manager.promote_types(to_promote_types)
    dst_block_table, dst_data_manager = compress_blocks(block_table, dst_data_manager)

    return type(df)(df._ctx, dst_block_table, df.partition_order_mappings, dst_data_manager)


def _isin(block_table, values, block_indexes):
    block_index_set = set(block_indexes)

    def _has_nan_value(v_list):
        for v in v_list:
            if np.isnan(v):
                return True

        return False

    if isinstance(values, list):

        def _is_in_list(blocks):
            ret_blocks = []
            for bid, block in enumerate(blocks):
                if bid not in block_index_set:
                    ret_blocks.append(block)
                elif isinstance(block, torch.Tensor):
                    ret_block = torch.isin(block, torch.Tensor(values))
                    if _has_nan_value(values):
                        ret_block |= torch.isnan(block)
                    ret_blocks.append(ret_block)
                elif isinstance(block, np.ndarray):
                    ret_block = np.isin(block, values)
                    if _has_nan_value(values):
                        ret_block |= np.isnan(block)
                    ret_blocks.append(torch.tensor(ret_block, dtype=torch.bool))

        block_table = block_table.mapValues(_is_in_list)
    else:

        def _is_in_dict(blocks):
            ret_blocks = []
            for bid, block in enumerate(blocks):
                if bid not in block_index_set:
                    ret_blocks.append(block)
                    continue
                elif isinstance(block, torch.Tensor):
                    ret_block = torch.zeros(block.shape, dtype=torch.bool)
                    for offset, in_values in values.get(bid, {}).items():
                        ret_block[:, offset] = torch.isin(block[:, offset], torch.Tensor(in_values))
                        if _has_nan_value(in_values):
                            ret_block[:, offset] |= torch.isnan(block[:, offset])
                    ret_blocks.append(ret_block)
                elif isinstance(block, np.ndarray):
                    ret_block = np.zeros(block.shape, dtype=np.bool_)
                    for offset, in_values in values.get(bid, {}).items():
                        ret_block[:, offset] = np.isin(block[:, offset], in_values)
                        if _has_nan_value(in_values):
                            ret_block[:, offset] = np.isnan(block[:, offset])
                    ret_blocks.append(torch.tensor(ret_block, dtype=torch.bool))

        block_table = block_table.mapValues(_is_in_dict)

    return block_table
