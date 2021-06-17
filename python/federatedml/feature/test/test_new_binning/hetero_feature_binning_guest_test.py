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

from arch.api import session
from fate_arch.computing import ComputingType
from fate_arch.session import computing_session as session
from federatedml.feature.hetero_feature_binning.hetero_binning_guest import HeteroFeatureBinningGuest
from federatedml.feature.instance import Instance


class TestHeteroFeatureBinning():
    def __init__(self):
        self.data_num = 10
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

        table = session.parallelize(final_result,
                                    include_key=True,
                                    partition=10)
        table.schema = {"header": self.header}
        self.model_name = 'HeteroFeatureBinning'

        self.table = table
        self.args = {"data": {self.model_name: {"data": table}}}

    def _make_param_dict(self, type='fit'):
        guest_componet_param = {
            "FeatureBinningParam": {
                "process_method": type,
            }
        }

        return guest_componet_param

    def test_feature_binning(self):
        binning_guest = HeteroFeatureBinningGuest()

        guest_param = self._make_param_dict('fit')

        binning_guest.run(guest_param, self.args)

        result_data = binning_guest.save_data()
        local_data = result_data.collect()
        print("data in fit")
        for k, v in local_data:
            print("k: {}, v: {}".format(k, v.features))
        guest_model = {self.model_name: binning_guest.export_model()}

        guest_args = {
            'data': {
                self.model_name: {
                    'data': self.table
                }
            },
            'model': guest_model
        }

        binning_guest = HeteroFeatureBinningGuest()

        guest_param = self._make_param_dict('transform')

        binning_guest.run(guest_param, guest_args)

        result_data = binning_guest.save_data()
        local_data = result_data.collect()
        print("data in transform")
        for k, v in local_data:
            print("k: {}, v: {}".format(k, v.features))

    def tearDown(self):
        self.table.destroy()


if __name__ == '__main__':
    import sys

    job_id = str(sys.argv[1])

    with Session() as session:
        session.init_computing(job_id, computing_type=ComputingType.STANDALONE)
        session.init_federation(job_id,
                                runtime_conf={"local": {
                                    "role": "guest",
                                    "party_id": 9999
                                },
                                    "role": {
                                        "host": [
                                            10000
                                        ],
                                        "guest": [
                                            9999
                                        ]
                                    }
                                })

        binning_obj = TestHeteroFeatureBinning()
        # homo_obj.test_homo_lr()
        binning_obj.test_feature_binning()
