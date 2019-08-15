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

from federatedml.logistic_regression.base_logistic_regression import BaseLogisticRegression
from federatedml.param import LogisticParam
from arch.api import eggroll
from federatedml.feature.instance import Instance
import numpy as np


class TestHomoLRGuest(unittest.TestCase):
    def setUp(self):
        # use default setting
        eggroll.init("123")
        logistic_param = LogisticParam()
        self.model = BaseLogisticRegression(logistic_param)
        self.model.header = []
        self.data_instance = self.__prepare_data()

    def __prepare_data(self, data_num=1000, feature_num=100):
        final_result = []
        for i in range(data_num):
            tmp = i * np.ones(feature_num)
            inst = Instance(inst_id=i, features=tmp, label=0)
            tmp = (i, inst)
            final_result.append(tmp)
        table = eggroll.parallelize(final_result,
                                    include_key=True,
                                    partition=3)
        return table

    # def test_save_load_model(self):
    #     n_iter_ = 10
    #     coef_ = [1., 0.2, 3.]
    #     intercept_ = 0.3
    #     classes_ = 2
    #
    #     model_table = "test_lr_table"
    #     model_namespace = "test_model_namesapce"
    #     self.model.save_model(model_table, model_namespace)
    #
    #     self.model.n_iter_ = n_iter_
    #     self.model.coef_ = coef_
    #     self.model.intercept_ = intercept_
    #     self.model.classes_ = classes_
    #
    #     # self.model.load_model(model_table=model_table, model_namespace=model_namespace)
    #     # Load model should change the value and make them not equal.
    #     #self.assertNotEqual(self.model.n_iter_, n_iter_)
    #     #self.assertNotEqual(self.model.coef_, coef_)
    #     #self.assertNotEqual(self.model.intercept_, intercept_)
    #     #self.assertNotEqual(self.model.classes_, classes_)
    #
    #     self.model.save_model(model_table, model_namespace)
    #     self.model.load_model(model_table, model_namespace)
    #     # self.assertEqual(self.model.n_iter_, n_iter_)
    #     self.assertEqual(self.model.coef_, coef_)
    #     self.assertEqual(self.model.intercept_, intercept_)
    #     # self.assertEqual(self.model.classes_, classes_)


if __name__ == '__main__':
    unittest.main()
