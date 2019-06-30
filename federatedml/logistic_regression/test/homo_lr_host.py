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


import numpy as np

from arch.api import eggroll
from arch.api import federation
from federatedml.logistic_regression.homo_logsitic_regression.homo_lr_host import HomoLRHost

from federatedml.feature.instance import Instance
np.random.seed(1)


class TestHomoLR(object):

    def __init__(self):
        self.data_num = 1000
        self.feature_num = 3
        self.header = ['x' + str(i) for i in range(self.feature_num)]
        final_result = []

        for i in range(self.data_num):
            tmp = []
            for _ in range(self.feature_num):
                tmp.append(np.random.choice([1, 2, 3]))
            tmp = np.array(tmp)
            inst = Instance(inst_id=i, features=tmp, label=i % 2)
            tmp_pair = (str(i), inst)
            final_result.append(tmp_pair)

        table = eggroll.parallelize(final_result,
                                    include_key=True,
                                    partition=10)
        table.schema = {"header": self.header}
        self.model_name = 'HomoLogisticRegression'

        self.table = table
        self.args = {"data": {self.model_name: {"train_data": table}}}

    def _make_param_dict(self):
        host_componet_param = {
            "LogisticParam": {
                "need_run": True,
                "cv_param": {
                    "need_cv": True,
                    "evaluate_param": {
                        "metrics": ["auc", "ks"]
                    }
                },
                "max_iter": 5
            }
        }

        return host_componet_param

    def test_homo_lr(self):
        homo_lr = HomoLRHost()

        host_param = self._make_param_dict()

        homo_lr.run(host_param, self.args)
        result_data = homo_lr.save_data()
        local_data = result_data.collect()
        print("data in fit")
        for k, v in local_data:
            print("k: {}, v: {}".format(k, v))

        lr_model = homo_lr.export_model()
        self.show_model(lr_model)

        host_model = {self.model_name: lr_model}

        host_args = {
            'data': {
                self.model_name: {
                    'eval_data': self.table
                }
            },
            'model': host_model
        }

        homo_lr = HomoLRHost()
        homo_lr.set_flowid('predict')

        host_param = self._make_param_dict()

        homo_lr.run(host_param, host_args)
        self.show_model(lr_model)

    def test_cv(self):
        homo_lr = HomoLRHost()
        guest_param = self._make_param_dict()
        homo_lr.run(guest_param, self.args)


    def show_model(self, model):
        meta_obj = model.get('HomoLogisticRegressionMeta')
        print("HomoLR meta info")
        print(meta_obj)

        param_obj = model.get('HomoLogisticRegressionParam')
        print("HomoLR param info")
        print(param_obj)

    def tearDown(self):
        self.table.destroy()


if __name__ == '__main__':
    import sys
    job_id = str(sys.argv[1])

    eggroll.init(job_id)
    federation.init(job_id,
                    {"local": {
                        "role": "host",
                        "party_id": 10000
                    },
                        "role": {
                            "host": [
                                10000
                            ],
                            "guest": [
                                9999
                            ],
                            "arbiter": [
                                10000
                            ]
                        }
                    })
    homo_obj = TestHomoLR()
    # homo_obj.test_homo_lr()
    homo_obj.test_cv()