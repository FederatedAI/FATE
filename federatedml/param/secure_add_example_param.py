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


class SecureAddExampleParam(BaseParam):
    def __init__(self, seed=None, partition=1, data_num=1000):
        self.seed = seed
        self.partition = partition
        self.data_num = data_num

    def check(self):
        if self.seed is not None and type(self.seed).__name__ != "int":
            raise ValueError("random seed should be None or integers")

        if type(self.partition).__name__ != "int" or self.partition < 1:
            raise ValueError("partition should be an integer large than 0")

        if type(self.data_num).__name__ != "int" or self.data_num < 1:
            raise ValueError("data_num should be an integer large than 0")

