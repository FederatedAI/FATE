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
from .._dataframe import DataFrame
from fate.arch.tensor.inside import GKSummary


def quantile(df: DataFrame, q, relative_error: float):
    if isinstance(q, float):
        q = [q]
    elif not isinstance(q, list):
        q = list(q)

    data_manager = df.data_manager
    column_names = data_manager.infer_operable_field_names()
    blocks_loc = [data_manager.loc_block(name) for name in column_names]

    def _mapper(blocks, columns_loc=None, error=None):
        column_size = len(columns_loc)
        gk_summary_obj_list = [GKSummary(error) for _ in range(column_size)]

        for idx, (bid, offset) in enumerate(columns_loc):
            gk_summary_obj_list[idx] += blocks[bid][:, offset]

        return gk_summary_obj_list

    def _reducer(l_gk_summary_obj_list, r_gk_summary_obj_list):
        rets = []
        for l_gk_summary_obj, r_gk_summary_obj in zip(l_gk_summary_obj_list, r_gk_summary_obj_list):
            rets.append(l_gk_summary_obj + r_gk_summary_obj)

        return rets

    gk_summary_func = functools.partial(_mapper, columns_loc=blocks_loc, error=relative_error)
    ret_gk_summary_obj_list = df.block_table.mapValues(gk_summary_func).reduce(_reducer)

    quantile_rets = dict()
    for column_name, gk_summary_obj in zip(column_names, ret_gk_summary_obj_list):
        query_ret = gk_summary_obj.queries(q)
        quantile_rets[column_name] = query_ret

    quantile_df = pd.DataFrame(quantile_rets, index=q)

    return quantile_df


def qcut(df: DataFrame, q: int):
    assert isinstance(q, int), f"to use qcut, {q} should be positive integer"
    max_ret = df.max()
    min_ret = df.min()

    dist = (max_ret - min_ret) / q

    cut_ret = []
    for i in range(1, q):
        cut_ret.append(min_ret + i * dist)

    cut_ret.append(max_ret)

    return pd.DataFrame(cut_ret, index=range(1, q + 1, 1))
