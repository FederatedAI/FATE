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


import unittest

# from arch.api import eggroll

# eggroll.init("123")

from federatedml.feature.feature_selection import UniqueValueFilter
from federatedml.param.feature_selection_param import UniqueValueParam


class TestFeatureSelect(unittest.TestCase):
    def setUp(self):
        param = UniqueValueParam()
        self.filter_obj = UniqueValueFilter(param, cols=-1)
        # self.filter_obj.left_cols = [0, 1]

    def test_protobuf(self):
        pass


if __name__ == '__main__':
    unittest.main()
