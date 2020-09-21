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

from federatedml.framework.homo.blocks import aggregator
from federatedml.framework.homo.test.blocks.test_utils import TestBlocks
from federatedml.util import consts


def aggregator_call(job_id, role, ind, *args):
    server_model = args[0][0]
    client_models = args[0][1:]
    if role == consts.ARBITER:
        agg = aggregator.Server()
        models = agg.get_models()
        agg.send_aggregated_model(server_model)
        return models
    else:
        agg = aggregator.Client()
        if role == consts.GUEST:
            agg.send_model(client_models[0])
        else:
            agg.send_model(client_models[ind + 1])
        return agg.get_aggregated_model()


class AggregatorTest(TestBlocks):

    def run_with_num_hosts(self, num_hosts):
        models = [np.random.rand(3, 4) for _ in range(num_hosts + 2)]
        server, *clients = self.run_test(aggregator_call, self.job_id, num_hosts, models)
        for model in clients:
            self.assertAlmostEqual(np.linalg.norm(model - models[0]), 0.0)
        for client_model, arbiter_get_model in zip(models[1:], server):
            self.assertAlmostEqual(np.linalg.norm(client_model - arbiter_get_model), 0.0)

    def test_host_1(self):
        self.run_with_num_hosts(1)

    def test_host_10(self):
        self.run_with_num_hosts(10)
