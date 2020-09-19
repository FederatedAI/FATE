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

import unittest

from federatedml.optim.initialize import Initializer
from federatedml.param import InitParam
from federatedml.util import consts
import numpy as np


class TestInitialize(unittest.TestCase):
    def test_initializer(self):
        initializer = Initializer()
        data_shape = 10
        init_param_obj = InitParam(init_method=consts.RANDOM_NORMAL,
                                   init_const=20,
                                   fit_intercept=False
                                   )
        model = initializer.init_model(model_shape=data_shape, init_params=init_param_obj)
        model_shape = np.array(model).shape
        self.assertTrue(model_shape == (10,))


if __name__ == '__main__':
    unittest.main()
