#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
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
import logging

from fate.arch import Context
from fate.arch.dataframe import DataFrame
from ..abc.module import Module

logger = logging.getLogger(__name__)


class Union(Module):
    def __init__(self, axis=0):
        self.axis = axis

    def fit(self, ctx: Context, train_data_list):
        sample_id_name, match_id_name = None, None
        if self.axis == 0:
            data_cols = set()
            for data in train_data_list:
                if sample_id_name:
                    if data.schema.sample_id_name != sample_id_name:
                        raise ValueError(f"Data sets should all have the same sample_id_name for union.")
                else:
                    sample_id_name = data.schema.sample_id_name
                if match_id_name:
                    if data.schema.match_id_name != match_id_name:
                        raise ValueError(f"Data sets should all have the same match_id_name for union.")
                else:
                    match_id_name = data.schema.match_id_name
                if data_cols:
                    if set(data.schema.columns) != data_cols:
                        raise ValueError(f"Data sets should all have the same columns for union on 0 axis.")
                else:
                    data_cols = set(data.schema.columns)
            result_data = DataFrame.vstack(train_data_list)
        elif self.axis == 1:
            col_set = set()
            for data in train_data_list:
                if sample_id_name:
                    if data.schema.sample_id_name != sample_id_name:
                        raise ValueError(f"Data sets should all have the same sample_id_name for union.")
                else:
                    sample_id_name = data.schema.sample_id_name
                if match_id_name:
                    if data.schema.match_id_name != match_id_name:
                        raise ValueError(f"Data sets should all have the same match_id_name for union.")
                else:
                    match_id_name = data.schema.match_id_name
                data_cols = set(data.schema.columns)
                if col_set.intersection(data_cols):
                    raise ValueError(f"column name conflict: {col_set.intersection(data_cols)}. "
                                     f"Please check input data")
                col_set.update(data_cols)
            result_data = DataFrame.hstack(train_data_list)
        else:
            raise ValueError(f"axis must be 0 or 1, but got {self.axis}")
        return result_data
