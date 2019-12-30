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

import os
import time
import unittest

# import numba
import numpy as np
import pandas as pd

from federatedml.util import fate_operator


# @numba.jit(nopython=True)
def go_fast(a):  # Function is compiled and runs in machine code
    sum = 0
    for j in range(100000):
        trace = 0
        for i in range(a.shape[0]):
            trace += np.tanh(a[i, i])
        sum += trace
    print(sum)
    return sum


class TestHomoLRGradient(unittest.TestCase):
    def setUp(self):
        home_dir = os.path.split(os.path.realpath(__file__))[0]
        data_dir = home_dir + '/../../../../examples/data/breast.csv'
        data_df = pd.read_csv(data_dir)
        self.X = np.array(data_df.iloc[:, 2:])
        self.Y = np.array(data_df.iloc[:, 1])

        self.Y = self.Y.reshape([-1, 1])
        self.coef = np.zeros(self.X.shape[1])
        self.intercept = 0
        self.fit_intercept = True

    def test_compute_time(self):
        x = np.arange(10000).reshape(100, 100)

        start_time = time.time()
        grad = self._test_compute(self.X, self.Y, self.coef, self.intercept, self.fit_intercept)
        # go_fast(x)
        end_time = time.time()
        print("compute time: {}".format(end_time - start_time))  # without jit: 6.935, with jit: 6.684
        # add jit in dot 7.271
        # add jit in dot only: 7.616
        pass

    # @numba.jit
    def _test_compute(self, X, Y, coef, intercept, fit_intercept):
        batch_size = len(X)
        if batch_size == 0:
            return None, None

        one_d_y = Y.reshape([-1, ])

        d = (0.25 * np.array(fate_operator.dot(X, coef) + intercept).transpose() + 0.5 * one_d_y * -1)
        grad_batch = X.transpose() * d

        tot_loss = np.log(1 + np.exp(np.multiply(-Y.transpose(), X.dot(coef) + intercept))).sum()
        avg_loss = tot_loss / Y.shape[0]

        # grad_batch = grad_batch.transpose()
        # if fit_intercept:
        #     grad_batch = np.c_[grad_batch, d]
        # grad = sum(grad_batch) / batch_size
        return 0




if __name__ == '__main__':
    unittest.main()
