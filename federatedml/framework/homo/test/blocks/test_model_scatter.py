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

import random

from federatedml.framework.homo.blocks import model_scatter
from federatedml.framework.homo.test.blocks.test_utils import TestBlocks
from federatedml.util import consts


def model_scatter_call(job_id, role, ind, *args):
    models = args[0]
    if role == consts.ARBITER:
        models = model_scatter.Server().get_models()
        return list(models)
    elif role == consts.HOST:
        model = models[ind + 1]
        return model_scatter.Client().send_model(model)
    else:
        model = models[0]
        return model_scatter.Client().send_model(model)


class ModelScatterTest(TestBlocks):

    def run_with_num_hosts(self, num_hosts):
        models = [[random.random() for _ in range(random.randint(1, 10))] for _ in range(num_hosts + 1)]
        arbiter, _, _ = self.run_test(model_scatter_call, self.job_id, num_hosts, models)

        for model, arbiter_model in zip(models, arbiter):
            self.assertListEqual(model, arbiter_model)

    def test_host_1(self):
        self.run_with_num_hosts(1)

    def test_host_10(self):
        self.run_with_num_hosts(10)
