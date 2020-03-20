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

from federatedml.framework.homo.blocks import model_broadcaster
from federatedml.framework.homo.test.blocks.test_utils import TestBlocks
from federatedml.util import consts


# noinspection PyUnusedLocal
def model_broadcaster_call(job_id, role, ind, *args):
    model_to_broadcast = args[0]
    if role == consts.ARBITER:
        return model_broadcaster.Server().send_model(model_to_broadcast)
    elif role == consts.HOST:
        return model_broadcaster.Client().get_model()
    else:
        return model_broadcaster.Client().get_model()


class ModelBroadcasterTest(TestBlocks):

    def run_with_num_hosts(self, num_hosts):
        import random
        model = [random.random() for _ in range(10)]

        arbiter, guest, hosts = self.run_test(model_broadcaster_call, self.job_id, num_hosts, model)
        self.assertListEqual(guest, model)
        for i in range(num_hosts):
            self.assertListEqual(hosts[i], model)

    def test_host_1(self):
        self.run_with_num_hosts(1)

    def test_host_10(self):
        self.run_with_num_hosts(10)
