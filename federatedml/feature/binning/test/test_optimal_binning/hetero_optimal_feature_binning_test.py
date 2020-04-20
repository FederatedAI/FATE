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

import argparse

import numpy as np

from arch.api import federation
from arch.api import session
from federatedml.feature.hetero_feature_binning.hetero_binning_guest import HeteroFeatureBinningGuest
from federatedml.feature.hetero_feature_binning.hetero_binning_host import HeteroFeatureBinningHost
from federatedml.feature.instance import Instance
from federatedml.feature.sparse_vector import SparseVector
from federatedml.param.feature_binning_param import FeatureBinningParam
from federatedml.util import consts

GUEST = 'guest'
HOST = 'host'


class TestHeteroFeatureBinning():
    def __init__(self, role, guest_id, host_id):
        self.role = role
        self.guest_id = guest_id
        self.host_id = host_id

        self.model_name = 'HeteroFeatureBinning'
        self.args = None
        self.table_list = []
        self.binning_obj = None

    def _gen_data(self, data_num, feature_num, partition, expect_ratio, is_sparse=False, use_random=False):
        data = []
        shift_iter = 0
        header = [str(i) for i in range(feature_num)]
        # bin_num = 3
        label_count = {}
        # expect_ratio = {
        #     0: (1, 9),
        #     1: (1, 1),
        #     2: (9, 1)
        # }
        bin_num = len(expect_ratio)

        for data_key in range(data_num):
            value = data_key % bin_num
            if value == 0:
                if shift_iter % bin_num == 0:
                    value = bin_num - 1
                shift_iter += 1
            if not is_sparse:
                if not use_random:
                    features = value * np.ones(feature_num)
                else:
                    features = np.random.random(feature_num)
                label = self.__gen_label(value, label_count, expect_ratio)
                inst = Instance(inst_id=data_key, features=features, label=label)

            else:
                if not use_random:
                    features = value * np.ones(feature_num)
                else:
                    features = np.random.random(feature_num)
                data_index = [x for x in range(feature_num)]
                sparse_inst = SparseVector(data_index, data=features, shape=10 * feature_num)
                label = self.__gen_label(value, label_count, expect_ratio)
                inst = Instance(inst_id=data_key, features=sparse_inst, label=label)
                header = [str(i) for i in range(feature_num * 10)]

            data.append((data_key, inst))
        result = session.parallelize(data, include_key=True, partition=partition)
        result.schema = {'header': header}
        self.table_list.append(result)
        return result

    def __gen_label(self, value, label_count: dict, expect_ratio: dict):
        """
        Generate label according to expect event and non-event ratio
        """
        if value not in expect_ratio:
            return np.random.randint(0, 2)

        expect_zero, expect_one = expect_ratio[value]
        if expect_zero == 0:
            return 1
        if expect_one == 0:
            return 0

        if value not in label_count:
            label = 1 if expect_one >= expect_zero else 0
            label_count[value] = [0, 0]
            label_count[value][label] += 1
            return label

        curt_zero, curt_one = label_count[value]
        if curt_zero == 0:
            label_count[value][0] += 1
            return 0
        if curt_one == 0:
            label_count[value][1] += 1
            return 1

        if curt_zero / curt_one <= expect_zero / expect_one:
            label_count[value][0] += 1
            return 0
        else:
            label_count[value][1] += 1
            return 1

    def _make_param_dict(self, process_type='fit'):
        guest_componet_param = {
            "local": {
                "role": self.role,
                "party_id": self.guest_id if self.role == GUEST else self.host_id
            },
            "role": {
                "guest": [
                    self.guest_id
                ],
                "host": [
                    self.host_id
                ]
            },
            "FeatureBinningParam": {
                "method": consts.OPTIMAL,
                "optimal_binning_param": {
                    "metric_method": "gini"
                }
            },
            "process_method": process_type,
        }
        return guest_componet_param

    def run_data(self, table_args, run_type='fit'):
        if self.binning_obj is not None:
            return self.binning_obj
        if self.role == GUEST:
            binning_obj = HeteroFeatureBinningGuest()
        else:
            binning_obj = HeteroFeatureBinningHost()

        # param_obj = FeatureBinningParam(method=consts.QUANTILE)
        # binning_obj.model_param = param_obj
        guest_param = self._make_param_dict(run_type)
        binning_obj.run(guest_param, table_args)
        print("current binning method: {}, split_points: {}".format(binning_obj.model_param.method,
                                                                    binning_obj.binning_obj.split_points))
        self.binning_obj = binning_obj
        return binning_obj

    def test_feature_binning(self):
        data_num = 1000
        feature_num = 50
        partition = 48
        expect_ratio = {
            0: (1, 9),
            1: (1, 1),
            2: (9, 1)
        }
        data_inst = self._gen_data(data_num, feature_num, partition, expect_ratio)
        table_args = {"data": {self.model_name: {"data": data_inst}}}
        self.args = table_args
        binning_obj = self.run_data(table_args, 'fit')

        result_data = binning_obj.save_data()
        fit_data = result_data.collect()
        fit_result = {}
        for k, v in fit_data:
            fit_result[k] = v.features
        fit_model = {self.model_name: binning_obj.export_model()}

        transform_args = {
            'data': {
                self.model_name: {
                    'data': data_inst
                }
            },
            'model': fit_model
        }

        # binning_guest = HeteroFeatureBinningGuest()
        transform_obj = self.run_data(transform_args, 'transform')

        # guest_param = self._make_param_dict('transform')

        # binning_guest.run(guest_param, guest_args)

        result_data = transform_obj.save_data()
        transformed_data = result_data.collect()
        print("data in transform")
        for k, v in transformed_data:
            fit_v: np.ndarray = fit_result.get(k)
            # print("k: {}, v: {}, fit_v: {}".format(k, v.features, fit_v))
            assert all(fit_v == v.features)
        return fit_model, transform_obj

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
    parser.add_argument('-j', '--job_id', required=True, type=str, help="job_id")

    args = parser.parse_args()
    job_id = args.job_id
    guest_id = args.gid
    host_id = args.hid
    role = args.role

    session.init(job_id)
    federation.init(job_id,
                    {"local": {
                        "role": role,
                        "party_id": guest_id if role == GUEST else host_id
                    },
                        "role": {
                            "host": [
                                host_id
                            ],
                            "guest": [
                                guest_id
                            ]
                        }
                    })

    test_obj = TestHeteroFeatureBinning(role, guest_id, host_id)
    # homo_obj.test_homo_lr()
    test_obj.test_feature_binning()
    test_obj.tearDown()
