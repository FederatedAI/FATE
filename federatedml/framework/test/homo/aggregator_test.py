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
from federatedml.framework.weights import ListVariables
from federatedml.util import consts
from .homo_test_sync_base import TestSyncBase


class AggregatorTest(TestSyncBase):
    @classmethod
    def call(cls, role, transfer_variable, ind, *args):
        weights = args[0]
        models = args[1]
        if role == consts.ARBITER:
            agg = aggregator.Arbiter()
            agg.register_aggregator(transfer_variable)
            agg.initialize_aggregator(True)
            agg.aggregate_and_broadcast()

        elif role == consts.HOST:
            agg = aggregator.Host()
            agg.register_aggregator(transfer_variable)
            agg.initialize_aggregator(weights[ind + 1])
            return agg.aggregate_and_get(ListVariables(list(models[ind + 1])))
        else:
            agg = aggregator.Guest()
            agg.register_aggregator(transfer_variable)
            agg.initialize_aggregator(weights[0])
            return agg.aggregate_and_get(ListVariables(list(models[0])))

    def run_with_num_hosts(self, num_hosts):
        import numpy as np
        import random
        weights = [random.random() for _ in range(num_hosts + 1)]
        total_weights = sum(weights)
        models = [np.random.rand(10) for _ in range(num_hosts + 1)]
        expert = list(
            np.sum([m * w / total_weights for m, w in zip(models, weights)],
                   0))

        arbiter, guest, *hosts = self.run_results(num_hosts, weights, models)
        guest = guest.parameters
        hosts = [host.parameters for host in hosts]
        for i in range(10):
            self.assertAlmostEqual(guest[i], expert[i])
            for host in hosts:
                self.assertAlmostEqual(host[i], expert[i])

    def test_host_1(self):
        self.run_with_num_hosts(1)

    def test_host_10(self):
        self.run_with_num_hosts(10)
