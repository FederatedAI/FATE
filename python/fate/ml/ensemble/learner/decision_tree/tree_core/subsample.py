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
from fate.arch.dataframe import DataFrame
import pandas as pd
import numpy as np


def goss_sampling(train_data: DataFrame, gh: DataFrame, top_rate=0.1, other_rate=0.1):
    sample_num = len(train_data)
    gh_df: pd.DataFrame = gh.as_pd_df()
    id_list = np.array(gh_df["sample_id"])
    g_arr = np.array(gh_df["g"]).astype(np.float64)
    h_arr = np.array(gh_df["h"]).astype(np.float64)

    g_sum_arr = np.abs(g_arr).sum(axis=1)  # if it is multi-classification case, we need to sum g
    abs_g_list_arr = g_sum_arr
    sorted_idx = np.argsort(-abs_g_list_arr, kind="stable")  # stable sample result

    a_part_num = int(sample_num * top_rate)
    b_part_num = int(sample_num * other_rate)

    if a_part_num == 0 or b_part_num == 0:
        raise ValueError("subsampled result is 0: top sample {}, other sample {}".format(a_part_num, b_part_num))

    # index of a part
    a_sample_idx = sorted_idx[:a_part_num]

    # index of b part
    rest_sample_idx = sorted_idx[a_part_num:]
    b_sample_idx = np.random.choice(rest_sample_idx, size=b_part_num, replace=False)

    # small gradient sample weights
    amplify_weights = (1 - top_rate) / other_rate
    g_arr[b_sample_idx] *= amplify_weights
    h_arr[b_sample_idx] *= amplify_weights

    # get selected sample
    a_idx_set, b_idx_set = set(list(a_sample_idx)), set(list(b_sample_idx))
    idx_set = a_idx_set.union(b_idx_set)
    selected_idx = np.array(list(idx_set))
    selected_g, selected_h = g_arr[selected_idx], h_arr[selected_idx]
    selected_id = id_list[selected_idx]

    new_gh = None
    subsample_data = None
