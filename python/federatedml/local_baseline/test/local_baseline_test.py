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

from fate_arch.common import profile
from fate_arch.session import computing_session as session
from federatedml.local_baseline.local_baseline import LocalBaseline
from federatedml.param.local_baseline_param import LocalBaselineParam
from federatedml.feature.instance import Instance
from sklearn.linear_model import LogisticRegression

profile._PROFILE_LOG_ENABLED = False


class TestLocalBaseline(unittest.TestCase):
    def setUp(self):
        self.job_id = str(uuid.uuid1())
        session.init("test_random_sampler_" + self.job_id)
        data_num = 100
        feature_num = 8
        self.prepare_data(data_num, feature_num)
        params = LocalBaselineParam()
        local_baseline_obj = LocalBaseline()
        local_baseline_obj._init_model(params)
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
            tmp = self.X[i, :]
            inst = Instance(inst_id=i, features=tmp, label=self.y[i])
            final_result.append((i, inst))
        table = session.parallelize(final_result,
                                    include_key=True,
                                    partition=3)
        self.table = table

    def test_predict(self):
        glm = LogisticRegression().fit(self.X, self.y)
        real_predict_result = glm.predict(self.X)
        real_predict_result = dict(zip(range(self.X.shape[0]), real_predict_result))
        self.local_baseline_obj.model_fit = glm
        model_predict_result = self.local_baseline_obj.predict(self.table)
        model_predict_result = {v[0]: v[1].features[1] for v in model_predict_result.collect()}

        self.assertDictEqual(model_predict_result, real_predict_result)

    def tearDown(self):
        session.stop()


if __name__ == '__main__':
    unittest.main()
