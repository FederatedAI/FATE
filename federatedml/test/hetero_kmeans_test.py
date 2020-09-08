# !/usr/bin/env python
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

import argparse

import numpy as np

from arch.api import federation
from arch.api import session

from federatedml.unsupervise.kmeans.hetero_kmeans.hetero_kmeans_arbiter import *
from federatedml.unsupervise.kmeans.hetero_kmeans.hetero_kmeans_client import *
from federatedml.unsupervise.kmeans.kmeans_model_base import *

from federatedml.feature.instance import Instance

GUEST = 'guest'
HOST = 'host'
Arbiter = 'arbiter'


class TestHeteroKmeans():
    def __init__(self, role, guest_id, host_id, arbiter_id):
        self.role = role
        self.guest_id = guest_id
        self.host_id = host_id
        self.arbiter_id = arbiter_id
        self.data_num = 100
        self.feature_num = 3
        self.partition =48
        # self.header = ['x' + str(i) for i in range(self.feature_num)]
        self.model_name = 'HeteroKmeans'
        # self.args = {"data": {self.model_name: {"data": table}}}
        self.args = None
        self.table_list = []
        self.kmeans_obj = None

    def _gen_data(self, data_num, feature_num, partition, is_sparse=False, use_random=False):
        header = [str(i) for i in range(feature_num)]
        final_result = []
        numpy_array = []
        for i in range(self.data_num):
            tmp = np.random.rand(self.feature_num)
            inst = Instance(inst_id=i, features=tmp, label=0)
            tmp_pair = (str(i), inst)
            final_result.append(tmp_pair)
            numpy_array.append(tmp)

        result = session.parallelize(final_result, include_key=True, partition=partition)
        result.schema = {'header': header}
        self.table_list.append(result)
        return result

    def _make_param_dict(self, process_type='fit'):
        guest_componet_param = {
            "local": {
                "role": self.role,
                "party_id": self.guest_id if self.role == GUEST else self.host_id if self.role == HOST else self.arbiter_id
            },
            "role": {
                "guest": [
                    self.guest_id
                ],
                "host": [
                    self.host_id
                ],
                "arbiter": [
                    self.arbiter_id
                ]

            },
            "process_method": process_type,
        }
        return guest_componet_param

    def run_data(self, table_args, run_type='fit'):
        if self.kmeans_obj is not None:
            return self.kmeans_obj
        if self.role == GUEST:
            kmeans_obj = HeteroKmeansGuest()
        elif self.role == HOST:
            kmeans_obj = HeteroKmeansHost()
        else:
            kmeans_obj = HeteroKmenasArbiter()
        guest_param = self._make_param_dict(run_type)

        kmeans_obj.run(guest_param, table_args)
        self.kmeans_obj = kmeans_obj
        return kmeans_obj

    def test_kmeans(self):
        data_num = 100
        feature_num = 3
        partition = 48
        data_inst = self._gen_data(data_num, feature_num, partition)
        table_args = {"data": {self.model_name: {"data": data_inst}}}
        self.args = table_args
        kmeans_obj = self.run_data(table_args, 'fit')

        # result_data = binning_obj.save_data()
        # fit_data = result_data.collect()
        #
        return kmeans_obj

    def tearDown(self):
        for table in self.table_list:
            table.destroy()
        print("Finish testing")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--role', required=False, type=str, help="role",
                        choices=(GUEST, HOST), default=GUEST)
    parser.add_argument('-gid', '--gid', required=False, type=str, help="guest party id", default='9999')
    parser.add_argument('-hid', '--hid', required=False, type=str, help="host party id", default='10000')
    parser.add_argument('-aid', '--aid', required=False, type=str, help="arbiter party id", default='10000')
    parser.add_argument('-j', '--job_id', required=True, type=str, help="job_id")

    args = parser.parse_args()
    job_id = args.job_id
    guest_id = args.gid
    host_id = args.hid
    arbiter_id = args.aid
    role = args.role

    session.init(job_id)
    federation.init(job_id,
                    {"local": {
                        "role": role,
                        "party_id": guest_id if role == GUEST else host_id if role == HOST else arbiter_id
                    },
                        "role": {
                            "host": [
                                host_id
                            ],
                            "guest": [
                                guest_id
                            ],
                            "arbiter":[
                                arbiter_id
                            ]
                        }
                    })

    test_obj = TestHeteroKmeans(role, guest_id, host_id, arbiter_id)
    # homo_obj.test_homo_lr()
    test_obj.test_kmeans()
    test_obj.tearDown()
    print('success')
