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


class FeatureUnion(Module):
    def __init__(self, axis=0):
        self.axis = axis

    def fit(self, ctx: Context, train_data_list):
        if self.axis == 0:
            result_data = DataFrame.vstack(train_data_list)
        elif self.axis == 1:
            col_set = set()
            for data in train_data_list:
                data_cols = set(data.schema.columns)
                if col_set.intersection(data_cols):
                    raise ValueError(f"column name conflict: {col_set.intersection(data_cols)}. "
                                     f"Please check input data")
                col_set.update(data_cols)
            result_data = DataFrame.hstack(train_data_list)
        else:
            raise ValueError(f"axis must be 0 or 1, but got {self.axis}")
        return result_data
