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
import numpy as np
import pandas as pd
import torch as t


def _process_dataframe(df):
    result_dict = {}

    for column in df.columns:
        unique_values = df[column].unique()
        sorted_values = sorted(unique_values)
        result_dict[column] = sorted_values

    return result_dict


def binning(data: DataFrame, max_bin=32):
    quantile = [i / max_bin for i in range(0, max_bin)]
    quantile_values = data.quantile(quantile)
    result_dict = _process_dataframe(quantile_values)

    return result_dict
