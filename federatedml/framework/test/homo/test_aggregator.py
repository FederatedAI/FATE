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

from federatedml.framework.homo.procedure import aggregator
from federatedml.framework.weights import OrderDictWeights
from federatedml.util import consts
from .homo_test_sync_base import TestSyncBase
import numpy as np
import random
import copy


class AggregatorTest(TestSyncBase):
    @classmethod
    def call(cls, role, transfer_variable, ind, *args):
        agg = aggregator.with_role(role, transfer_variable, enable_secure_aggregate=True)
        if role == consts.ARBITER:
            agg.aggregate_and_broadcast()
            print(agg.aggregate_loss())
        else:
            # disorder dit
            order = list(range(5))
            np.random.seed(random.SystemRandom().randint(1, 100))
            np.random.shuffle(order)
            raw = {k: np.random.rand(10, 10) for k in order}

            w = OrderDictWeights(copy.deepcopy(raw))
            d = random.random()
            aggregated = agg.aggregate_then_get(w, degree=d)

            agg.send_loss(2.0)
            return aggregated, raw, d

    def run_with_num_hosts(self, num_hosts):
        _, guest, *hosts = self.run_results(num_hosts)
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
