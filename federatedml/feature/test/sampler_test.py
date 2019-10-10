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
import random
import unittest

from arch.api import session
from fate_flow.manager.tracking import Tracking 
from federatedml.feature.instance import Instance
from federatedml.feature.sampler import RandomSampler
from federatedml.feature.sampler import StratifiedSampler
from federatedml.util import consts


class TestRandomSampler(unittest.TestCase):
    def setUp(self):
        session.init("test_random_sampler")
        self.data = [(i * 10 + 5, i * i) for i in range(100)]
        self.table = session.parallelize(self.data, include_key=True)
        self.data_to_trans = [(i * 10 + 5, i * i * i) for i in range(100)]
        self.table_trans = session.parallelize(self.data_to_trans, include_key=True)

    def test_downsample(self):
        sampler = RandomSampler(fraction=0.3, method="downsample")
        tracker = Tracking("jobid", "guest", 9999, "abc", "123")
        sampler.set_tracker(tracker)
        sample_data, sample_ids = sampler.sample(self.table)
        
        self.assertTrue(sample_data.count() > 25 and sample_data.count() < 35)
        self.assertTrue(len(set(sample_ids)) == len(sample_ids))
        
        new_data = list(sample_data.collect())
        data_dict = dict(self.data)
        for id, value in new_data:
            self.assertTrue(id in data_dict)
            self.assertTrue(np.abs(value - data_dict.get(id)) < consts.FLOAT_ZERO)

        trans_sampler = RandomSampler(method="downsample")
        trans_sampler.set_tracker(tracker)
        trans_sample_data = trans_sampler.sample(self.table_trans, sample_ids)
        trans_data = list(trans_sample_data.collect())
        trans_sample_ids = [id for (id, value) in trans_data]
        data_to_trans_dict = dict(self.data_to_trans)
        sample_id_mapping = dict(zip(sample_ids, range(len(sample_ids))))
        
        self.assertTrue(len(trans_data) == len(sample_ids))
        self.assertTrue(set(trans_sample_ids) == set(sample_ids))

        for id, value in trans_data:
            self.assertTrue(id in sample_id_mapping)
            self.assertTrue(np.abs(value - data_to_trans_dict.get(id)) < consts.FLOAT_ZERO)

    def test_upsample(self):
        sampler = RandomSampler(fraction=3, method="upsample")
        tracker = Tracking("jobid", "guest", 9999, "abc", "123")
        sampler.set_tracker(tracker)
        sample_data, sample_ids = sampler.sample(self.table)

        self.assertTrue(sample_data.count() > 250 and sample_data.count() < 350)
        
        data_dict = dict(self.data)
        new_data = list(sample_data.collect())
        for id, value in new_data:
            self.assertTrue(np.abs(value - data_dict[sample_ids[id]]) < consts.FLOAT_ZERO)

        trans_sampler = RandomSampler(method="upsample")
        trans_sampler.set_tracker(tracker)
        trans_sample_data = trans_sampler.sample(self.table_trans, sample_ids)
        trans_data = list(trans_sample_data.collect())
        data_to_trans_dict = dict(self.data_to_trans)
        
        self.assertTrue(len(trans_data) == len(sample_ids))
        for id, value in trans_data:
            self.assertTrue(np.abs(value - data_to_trans_dict[sample_ids[id]]) < consts.FLOAT_ZERO)


class TestStratifiedSampler(unittest.TestCase):
    def setUp(self):
        session.init("test_stratified_sampler")
        self.data = []
        self.data_to_trans = []
        for i in range(1000):
            self.data.append((i, Instance(label=i % 4, features=i * i)))
            self.data_to_trans.append((i, Instance(features = i ** 3)))

        self.table = session.parallelize(self.data, include_key=True)
        self.table_trans = session.parallelize(self.data_to_trans, include_key=True)

    def test_downsample(self):
        fractions = [(0, 0.3), (1, 0.4), (2, 0.5), (3, 0.8)]
        sampler = StratifiedSampler(fractions=fractions, method="downsample")
        tracker = Tracking("jobid", "guest", 9999, "abc", "123")
        sampler.set_tracker(tracker)
        sample_data, sample_ids = sampler.sample(self.table)
        count_label = [0 for i in range(4)]
        new_data = list(sample_data.collect())
        data_dict = dict(self.data)
        self.assertTrue(set(sample_ids) & set(data_dict.keys()) == set(sample_ids))

        for id, inst in new_data:
            count_label[inst.label] += 1
            self.assertTrue(type(id).__name__ == 'int' and id >= 0 and id < 1000)
            self.assertTrue(inst.label == self.data[id][1].label and inst.features == self.data[id][1].features)

        for i in range(4):
            self.assertTrue(np.abs(count_label[i] - 250 * fractions[i][1]) < 10)

        trans_sampler = StratifiedSampler(method="downsample")
        trans_sampler.set_tracker(tracker)
        trans_sample_data = trans_sampler.sample(self.table_trans, sample_ids)
        trans_data = list(trans_sample_data.collect())
        trans_sample_ids = [id for (id, value) in trans_data]
        data_to_trans_dict = dict(self.data_to_trans)
     
        self.assertTrue(set(trans_sample_ids) == set(sample_ids))
        for id, inst in trans_data:
            self.assertTrue(inst.features == data_to_trans_dict.get(id).features)
     
    def test_upsample(self):
        fractions = [(0, 1.3), (1, 0.5), (2, 0.8), (3, 9)]
        sampler = StratifiedSampler(fractions=fractions, method="upsample")
        tracker = Tracking("jobid", "guest", 9999, "abc", "123")
        sampler.set_tracker(tracker)
        sample_data, sample_ids = sampler.sample(self.table)
        new_data = list(sample_data.collect())
        count_label = [0 for i in range(4)]
        data_dict = dict(self.data)

        for id, inst in new_data:
            count_label[inst.label] += 1
            self.assertTrue(type(id).__name__ == 'int' and id >= 0 and id < len(sample_ids))
            real_id = sample_ids[id]
            self.assertTrue(inst.label == self.data[real_id][1].label and
                            inst.features == self.data[real_id][1].features)

        for i in range(4):
            self.assertTrue(np.abs(count_label[i] - 250 * fractions[i][1]) < 10)
        
        trans_sampler = StratifiedSampler(method="upsample")
        trans_sampler.set_tracker(tracker)
        trans_sample_data = trans_sampler.sample(self.table_trans, sample_ids)
        trans_data = (trans_sample_data.collect())
        trans_sample_ids = [id for (id, value) in trans_data]
        data_to_trans_dict = dict(self.data_to_trans)

        self.assertTrue(sorted(trans_sample_ids) == list(range(len(sample_ids))))
        for id, inst in trans_data:
            real_id = sample_ids[id]
            self.assertTrue(inst.features == self.data_to_trans_dict[real_id][1].features)


if __name__ == '__main__':
    unittest.main()
