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

import math
import unittest

from federatedml.optim.convergence import converge_func_factory


class TestConvergeFunction(unittest.TestCase):
    def test_diff_converge(self):
        loss = 50
        eps = 0.00001
        # converge_func = DiffConverge(eps=eps)
        converge_func = converge_func_factory(early_stop='diff', tol=eps)
        iter_num = 0
        pre_loss = loss
        while iter_num < 500:
            loss *= 0.5
            converge_flag = converge_func.is_converge(loss)
            if converge_flag:
                break
            iter_num += 1
            pre_loss = loss
        self.assertTrue(math.fabs(pre_loss - loss) <= eps)

    def test_abs_converge(self):
        loss = 50
        eps = 0.00001
        # converge_func = AbsConverge(eps=eps)
        converge_func = converge_func_factory(early_stop='abs', tol=eps)

        iter_num = 0
        while iter_num < 500:
            loss *= 0.5
            converge_flag = converge_func.is_converge(loss)
            if converge_flag:
                break
            iter_num += 1
        self.assertTrue(math.fabs(loss) <= eps)


if __name__ == '__main__':
    unittest.main()
