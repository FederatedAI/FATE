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

import time
import unittest

import numpy as np

from arch.api import session
from federatedml.feature.instance import Instance
from federatedml.model_selection import MiniBatch
from federatedml.model_selection import indices

session.init("123")


class TestMiniBatch(unittest.TestCase):
    def prepare_data(self, data_num, feature_num):
        final_result = []
        for i in range(data_num):
            tmp = i * np.ones(feature_num)
            inst = Instance(inst_id=i, features=tmp, label=0)
            tmp = (i, inst)
            final_result.append(tmp)
        table = session.parallelize(final_result,
                                    include_key=True,
                                    partition=3)
        return table

    def test_mini_batch_data_generator(self, data_num=100, batch_size=320):
        t0 = time.time()
        feature_num = 20

        expect_batches = data_num // batch_size
        # print("expect_batches: {}".format(expect_batches))
        data_instances = self.prepare_data(data_num=data_num, feature_num=feature_num)
        # print("Prepare data time: {}".format(time.time() - t0))
        mini_batch_obj = MiniBatch(data_inst=data_instances, batch_size=batch_size)
        batch_data_generator = mini_batch_obj.mini_batch_data_generator()
        batch_id = 0
        pre_time = time.time() - t0
        # print("Prepare mini batch time: {}".format(pre_time))
        total_num = 0
        for batch_data in batch_data_generator:
            batch_num = batch_data.count()
            if batch_id < expect_batches - 1:
                # print("In mini batch test, batch_num: {}, batch_size:{}".format(
                #     batch_num, batch_size
                # ))
                self.assertEqual(batch_num, batch_size)

            batch_id += 1
            total_num += batch_num
            # curt_time = time.time()
            # print("One batch time: {}".format(curt_time - pre_time))
            # pre_time = curt_time
        self.assertEqual(total_num, data_num)

    def test_collect_index(self):
        data_num = 100
        feature_num = 20
        data_instances = self.prepare_data(data_num=data_num, feature_num=feature_num)
        # res = data_instances.mapValues(lambda x: x)
        data_sids_iter, data_size = indices.collect_index(data_instances)
        self.assertEqual(data_num, data_size)
        real_index_num = 0
        for sid, _ in data_sids_iter:
            real_index_num += 1
        self.assertEqual(data_num, real_index_num)

    def test_data_features(self):
        data_num = 100
        feature_num = 20
        data_instances = self.prepare_data(data_num=data_num, feature_num=feature_num)
        local_data = data_instances.collect()
        idx, data = local_data.__next__()
        features = data.features
        self.assertEqual(len(features), feature_num)

    def test_different_datasize_batch(self):
        data_nums = [10, 100]
        batch_size = [1, 2, 10, 32]

        for d_n in data_nums:
            for b_s in batch_size:
                # print("data_nums: {}, batch_size: {}".format(d_n, b_s))
                self.test_mini_batch_data_generator(data_num=d_n, batch_size=b_s)


if __name__ == '__main__':
    unittest.main()
