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
from arch.api import federation
from federatedml.feature.hetero_feature_binning.hetero_binning_host import HeteroFeatureBinningHost
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
        host_componet_param = {
            "FeatureBinningParam": {
                "process_method": type,
            }
        }
        return host_componet_param

    def test_feature_binning(self):
        binning_host = HeteroFeatureBinningHost()

        host_param = self._make_param_dict('fit')

        binning_host.run(host_param, self.args)

        result_data = binning_host.save_data()
        local_data = result_data.collect()
        print("data in fit")
        for k, v in local_data:
            print("k: {}, v: {}".format(k, v.features))

        host_model = {self.model_name: binning_host.export_model()}

        host_args = {
            'data': {
                self.model_name: {
                    'data': self.table
                }
            },
            'model': host_model
        }

        binning_host = HeteroFeatureBinningHost()

        host_param = self._make_param_dict('transform')

        binning_host.run(host_param, host_args)

        result_data = binning_host.save_data()
        local_data = result_data.collect()
        print("data in transform")
        for k, v in local_data:
            print("k: {}, v: {}".format(k, v.features))

    def tearDown(self):
        self.table.destroy()


if __name__ == '__main__':
    import sys
    job_id = str(sys.argv[1])

    session.init(job_id)
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
                            ]
                        }
                    })
    binning_obj = TestHeteroFeatureBinning()
    # homo_obj.test_homo_lr()
    binning_obj.test_feature_binning()