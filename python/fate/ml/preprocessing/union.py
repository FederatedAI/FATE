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
        sample_id_name_list = [data.schema.sample_id_name for data in train_data_list]
        if sum([name != sample_id_name_list[0] for name in sample_id_name_list]):
            raise ValueError(f"Data sets should all have the same sample_id_name for union.")

        match_id_name_list = [data.schema.match_id_name for data in train_data_list]
        if sum([name != match_id_name_list[0] for name in match_id_name_list]):
            raise ValueError(f"Data sets should all have the same match_id_name for union.")

        if self.axis == 0:
            label_name_list = [data.schema.label_name for data in train_data_list]
            if sum([name != label_name_list[0] for name in label_name_list]):
                raise ValueError(f"Data sets should all have the same label_name for union.")

            column_name_list = [set(data.schema.columns) for data in train_data_list]
            if sum([col_names != column_name_list[0] for col_names in column_name_list]):
                raise ValueError(f"Data sets should all have the same columns for union on 0 axis.")
            result_data = DataFrame.vstack(train_data_list)
            return result_data
        else:
            raise ValueError(f"axis must be 0, but got {self.axis}")
