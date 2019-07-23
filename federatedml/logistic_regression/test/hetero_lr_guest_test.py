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
from federatedml.logistic_regression.hetero_logistic_regression import HeteroLRGuest
# from federatedml.logistic_regression.homo_logsitic_regression.homo_lr_guest import HomoLRGuest
from federatedml.feature.instance import Instance

np.random.seed(1)


class TestHeteroLR(object):
    def __init__(self):
        guest_data_file = "../../../examples/data/breast_b.csv"
        self.model_name = 'HeteroLogisticRegression'
        self.table = None
        self.args = self.generate_args(guest_data_file)

    def generate_args(self, input_data_file, has_header=1):
        header, data = self.read_data(input_data_file, has_header)

        inst_pairs = []
        for i, line in enumerate(data):
            inst = Instance(inst_id=line[0], features=np.array(line[2:], dtype=float), label=line[1])
            inst_pair = (str(line[0]), inst)
            inst_pairs.append(inst_pair)

        self.table = eggroll.parallelize(inst_pairs,
                                         include_key=True,
                                         partition=10)
        self.table.schema = {"header": header[2:]}

        return {"data": {self.model_name: {"train_data": self.table, "eval_data": self.table}}}

    def read_data(self, input_file, has_header):
        with open(input_file, "r") as fin:
            header = None
            if has_header:
                header = fin.readline().replace("\n", "").split(",")

            data = []
            line = fin.readline()
            while True:
                if not line:
                    break
                else:
                    data.append(line.replace("\n", "").split(","))
                    line = fin.readline()

            return header, data

    def _make_param_dict(self):
        guest_componet_param = {
            "LogisticParam": {
                "max_iter": 2
            }
        }
        return guest_componet_param

    def test_hetero_lr(self):
        hetero_lr_obj = HeteroLRGuest()
        hetero_lr_obj.set_flowid('train')
        guest_param = self._make_param_dict()

        hetero_lr_obj.run(guest_param, self.args)
        result_data = hetero_lr_obj.save_data()
        local_data = result_data.collect()
        print("data in fit")
        for k, v in local_data:
            print("k: {}, v: {}".format(k, v))

        lr_model = hetero_lr_obj.export_model()
        self.show_model(lr_model)

        guest_model = {self.model_name: lr_model}

        guest_args = {
            'data': {
                self.model_name: {
                    'eval_data': self.table
                }
            },
            'model': guest_model
        }

        hetero_lr_obj = HeteroLRGuest()
        hetero_lr_obj.set_flowid('predict')
        guest_param = self._make_param_dict()

        print("guest_args:{}".format(guest_args))
        hetero_lr_obj.run(guest_param, guest_args)
        self.show_model(lr_model)
        result_data = hetero_lr_obj.save_data()
        local_data = result_data.collect()
        print("data in transform")
        for k, v in local_data:
            print("k: {}, v: {}".format(k, v))

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
                        "role": "guest",
                        "party_id": 9999
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
    # unittest.main()
    test_obj = TestHeteroLR()
    test_obj.test_hetero_lr()
