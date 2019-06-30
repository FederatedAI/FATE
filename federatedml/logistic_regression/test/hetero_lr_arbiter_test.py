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
from federatedml.logistic_regression.hetero_logistic_regression import HeteroLRArbiter

np.random.seed(1)


class TestHeteroLR(object):

    def __init__(self):
        self.model_name = 'HeteroLogisticRegression'
        self.args = {"data": None}

    def _make_param_dict(self):
        arbiter_componet_param = {
            "LogisticParam": {
                "max_iter": 2
            }
        }
        return arbiter_componet_param

    def test_hetero_lr(self):
        hetero_lr_obj = HeteroLRArbiter()
        hetero_lr_obj.set_flowid('train')
        arbiter_param = self._make_param_dict()
        hetero_lr_obj.run(arbiter_param, self.args)

        lr_model = hetero_lr_obj.export_model()
        self.show_model(lr_model)

        arbiter_model = {self.model_name: lr_model}
        arbiter_args = {
            'data': {
                self.model_name: {
                    'eval_data': None
                }
            },
            'model': arbiter_model
        }

        hetero_lr_obj = HeteroLRArbiter()
        hetero_lr_obj.set_flowid('predict')
        arbiter_param = self._make_param_dict()

        hetero_lr_obj.run(arbiter_param, arbiter_args)
        self.show_model(lr_model)

    # def test_cv(self):
    #     homo_lr = HomoLRArbiter()
    #     guest_param = self._make_param_dict()
    #     homo_lr.run(guest_param, self.args)

    def show_model(self, model):
        meta_obj = model.get('HeteroLogisticRegressionMeta')
        print("HeteroLR meta info")
        print(meta_obj)

        param_obj = model.get('HeteroLogisticRegressionParam')
        print("HeteroLR param info")
        print(param_obj)


if __name__ == '__main__':
    from federatedml.logistic_regression.test.job_id import job_id

    eggroll.init(job_id)
    federation.init(job_id,
                    {"local": {
                        "role": "arbiter",
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
    test_obj = TestHeteroLR()
    test_obj.test_hetero_lr()
