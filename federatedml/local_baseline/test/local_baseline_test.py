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

import numpy as np
import unittest
import uuid

from arch.api import session
from federatedml.feature.instance import Instance
from federatedml.local_baseline.local_baseline import LocalBaseline
from sklearn.linear_model import LogisticRegression

class TestLocalBaselin(unittest.TestCase):
    def setUp(self):
        self.job_id = str(uuid.uuid1())
        session.init(self.job_id)
        data_num = 100
        feature_num = 8
        self.prepare_data(data_num, feature_num)
        local_baseline_obj = LocalBaseline()
        local_baseline_obj.need_run = True
        local_baseline_obj.header = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"]
        local_baseline_obj.model_name = "LogisticRegression"
        local_baseline_obj.model_opts = {}
        self.local_baseline_obj = local_baseline_obj

    def prepare_data(self, data_num, feature_num):
        self.X = np.random.randint(0, 10, (data_num, feature_num))
        self.y = np.random.randint(0, 2, data_num)
        final_result = []
        for i in range(data_num):
            tmp = self.X[i,:]
            inst = Instance(inst_id=i, features=tmp, label=self.y[i])
            tmp = (str(i), inst)
            final_result.append(tmp)
        table = session.parallelize(final_result,
                                    include_key=True,
                                    partition=3)
        self.table = table

    def test_predict(self):
        glm = LogisticRegression().fit(self.X, self.y)
        real_predict_result = glm.predict(self.X)
        self.local_baseline_obj.model_fit = glm
        model_predict_result = self.local_baseline_obj.predict(self.table)
        model_predict_result = np.array([v[1][1] for v in model_predict_result.collect()])

        np.testing.assert_array_equal(model_predict_result, real_predict_result)

    def tearDown(self):
         session.stop()
         try:
             session.cleanup("*", self.job_id, True)
         except EnvironmentError:
             pass
         try:
             session.cleanup("*", self.job_id, False)
         except EnvironmentError:
             pass


if __name__ == '__main__':
    unittest.main()
