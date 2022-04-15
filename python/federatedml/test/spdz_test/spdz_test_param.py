#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from federatedml.param.base_param import BaseParam


class SPDZTestParam(BaseParam):
    def __init__(self, data_num=10000, test_round=1, data_partition=4, seed=123,
                 data_lower_bound=-1000000000, data_upper_bound=1000000000):
        self.data_num = data_num
        self.test_round = test_round
        self.seed = seed
        self.data_partition = data_partition
        self.data_lower_bound = data_lower_bound
        self.data_upper_bound = data_upper_bound

    def check(self):
        if self.seed is None or not isinstance(self.seed, int):
            raise ValueError("random seed should be integer")

        if not isinstance(self.test_round, int) or self.test_round < 1:
            raise ValueError("test_round should be positive integer")

        if not isinstance(self.data_num, int) or self.data_num < 1:
            raise ValueError("data_num should be positive integer")

        if not isinstance(self.data_partition, int) or self.data_partition < 1:
            raise ValueError("data partition should be positive integer")

        if not isinstance(self.data_upper_bound, (int, float)) or not isinstance(self.data_lower_bound, (int, float)):
            raise ValueError("bound of data should be numeric")
