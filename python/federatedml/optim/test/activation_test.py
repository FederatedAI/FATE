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

import math
import unittest
import numpy as np
from federatedml.optim import activation


class TestConvergeFunction(unittest.TestCase):
    def test_numeric_stability(self):
        x_list = np.linspace(-709, 709, 10000)

        # Original function
        # a = 1. / (1. + np.exp(-x))
        for x in x_list:
            a1 = 1. / (1. + np.exp(-x))
            a2 = activation.sigmoid(x)
            self.assertTrue(np.abs(a1 - a2) < 1e-5)


if __name__ == '__main__':
    unittest.main()
