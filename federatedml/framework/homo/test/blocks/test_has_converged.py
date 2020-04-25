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

from federatedml.framework.homo.blocks import has_converged
from federatedml.framework.homo.test.blocks.test_utils import TestBlocks
from federatedml.util import consts


# noinspection PyUnusedLocal
def model_broadcaster_call(job_id, role, ind, *args):
    status = args[0]
    if role == consts.ARBITER:
        return has_converged.Server().remote_converge_status(status)
    else:
        return has_converged.Client().get_converge_status()


class ModelBroadcasterTest(TestBlocks):

    def run_with_num_hosts(self, num_hosts):
        status = random.random() > 0.5

        arbiter, guest, hosts = self.run_test(model_broadcaster_call, self.job_id, num_hosts, status)
        self.assertEqual(guest, status)
        for i in range(num_hosts):
            self.assertEqual(hosts[i], status)

    def test_host_1(self):
        self.run_with_num_hosts(1)

    def test_host_10(self):
        self.run_with_num_hosts(10)
