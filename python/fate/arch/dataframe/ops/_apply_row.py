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

from collections import Iterable

from .._dataframe import DataFrame
from ..manager.data_manager import DataManager


def apply_row(df: "DataFrame", func, result_type="expand",
              columns: list=None, with_label=False, with_weight=False) -> "DataFrame":
    """
    In current version, assume that the apply_row results' lengths are equal
    """
    data_manager = df.data_manager
    dst_data_manager, _ = data_manager.derive_new_data_manager(with_sample_id=True,
                                                            with_match_id=True,
                                                            with_label=not with_weight,
                                                            with_weight=not with_weight,
                                                            columns=None)

    non_operable_field_names = dst_data_manager.get_field_name_list()
    non_operable_fields_loc = [data_manager.loc_block(field_name) for field_name in non_operable_field_names]

    fields_loc = data_manager.get_fields_loc(with_label=with_label, with_weight=with_weight)
    fields_name = data_manager.get_field_name_list(with_sample_id=False,
                                                   with_match_id=False,
                                                   with_label=with_label,
                                                   with_weight=with_weight)

    _apply_func = functools.partial(_apply, func=func, result_type=result_type, src_field_names=fields_name,
                                    src_fields_loc=fields_loc, src_non_operable_fields_loc=non_operable_fields_loc,
                                    ret_columns=columns, dst_dm=dst_data_manager)

    dst_block_table = df.block_table.mapValues(_apply)

    return DataFrame(
        df._ctx,
        dst_block_table,
        df.partition_order_mappings,
        dst_data_manager
    )


def _apply(blocks, func=None, result_type=None, src_field_names=None,
            src_fields_loc=None, src_non_operable_fields_loc=None, ret_columns=None, dst_dm: "DataManager"=None):
    dm = dst_dm.duplicate()
    ret_blocks = []
    lines = len(blocks[0])
    ret_column_len = len(ret_columns) if ret_columns is not None else None
    reserved_blocks = []

    for lid in range(lines):
        apply_row_data = []
        for bid, offset in src_fields_loc:
            if isinstance(blocks[bid], torch.Tensor):
                apply_row_data.append(blocks[bid][lid][bid].item())
            else:
                apply_row_data.append(blocks[bid][lid][bid])

        apply_ret = func(pd.Series(apply_row_data, index=src_field_names))

        if isinstance(apply_ret, Iterable):
            apply_ret = list(apply_ret)
            if ret_column_len is None:
                ret_column_len = len(apply_ret)
            elif ret_column_len != len(apply_ret):
                raise ValueError("Result of apply row should have equal length")
        else:
            if ret_column_len and ret_column_len != 1:
                raise ValueError("Result of apply row should have equal length")
            apply_ret = [apply_ret]

    return ret_blocks, dst_dm
