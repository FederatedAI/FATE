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

from federatedml.framework.homo.blocks import loss_scatter
from federatedml.framework.homo.test.blocks.test_utils import TestBlocks
from federatedml.util import consts


def loss_scatter_call(job_id, role, ind, *args):
    losses = args[0]
    if role == consts.ARBITER:
        losses = loss_scatter.Server().get_losses()
        return list(losses)
    elif role == consts.HOST:
        loss = losses[ind + 1]
        return loss_scatter.Client().send_loss(loss)
    else:
        loss = losses[0]
        return loss_scatter.Client().send_loss(loss)


class LossScatterTest(TestBlocks):

    def run_with_num_hosts(self, num_hosts):
        losses = [random.random() for _ in range(num_hosts + 1)]
        arbiter, _, _ = self.run_test(loss_scatter_call, self.job_id, num_hosts, losses)

        for loss, arbiter_got_loss in zip(losses, arbiter):
            self.assertEqual(loss, arbiter_got_loss)

    def test_host_1(self):
        self.run_with_num_hosts(1)

    def test_host_10(self):
        self.run_with_num_hosts(10)
