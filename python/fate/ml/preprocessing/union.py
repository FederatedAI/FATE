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

from fate.interface import Context
from fate.interface import Dataframe
from ..abc.module import Module

logger = logging.getLogger(__name__)


class Union(Module):
    def __init__(self, axis=0):
        self.axis = axis

    def fit(self, ctx: Context, train_data_list, validate_data=None) -> None:
        if self.axis == 0:
            result_data = Dataframe.vstack(train_data_list)
        elif self.axis == 1:
            result_data = Dataframe.hstack(train_data_list)
        else:
            raise ValueError(f"axis must be 0 or 1, but got {self.axis}")
        return result_data
