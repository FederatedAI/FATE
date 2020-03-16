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

import copy
import numpy as np
import random

from federatedml.framework.homo.blocks import secure_mean_aggregator
from federatedml.framework.homo.test.blocks.test_utils import TestBlocks
from federatedml.framework.weights import OrderDictWeights
from federatedml.util import consts


# noinspection PyUnusedLocal
def secure_aggregator_call(job_id, role, ind, *args):
    if role == consts.ARBITER:
        agg = secure_mean_aggregator.Server()
        model = agg.weighted_mean_model()
        agg.send_aggregated_model(model)
    else:
        agg = secure_mean_aggregator.Client()
        # disorder dit
        order = list(range(5))
        np.random.seed(random.SystemRandom().randint(1, 100))
        np.random.shuffle(order)
        raw = {k: np.random.rand(10, 10) for k in order}

        w = OrderDictWeights(copy.deepcopy(raw))
        d = random.random()
        agg.send_weighted_model(w, weight=d)
        aggregated = agg.get_aggregated_model()
        return aggregated, raw, d


class AggregatorTest(TestBlocks):

    def run_with_num_hosts(self, num_hosts):
        _, guest, hosts = self.run_test(secure_aggregator_call, self.job_id, num_hosts)
        expert = OrderDictWeights(guest[1]) * guest[2]
        total_weights = guest[2]
        aggregated = [guest[0]]
        for host in hosts:
            expert += OrderDictWeights(host[1]) * host[2]
            total_weights += host[2]
            aggregated.append(host[0])
        expert /= total_weights
        expert = expert.unboxed
        aggregated = [w.unboxed for w in aggregated]

        for k in expert:
            for w in aggregated:
                self.assertAlmostEqual(np.linalg.norm(expert[k] - w[k]), 0.0)

    def test_host_1(self):
        self.run_with_num_hosts(1)

    def test_host_10(self):
        self.run_with_num_hosts(10)
