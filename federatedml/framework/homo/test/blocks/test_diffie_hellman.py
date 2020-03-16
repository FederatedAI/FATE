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

from federatedml.framework.homo.blocks import uuid_generator, diffie_hellman
from federatedml.framework.homo.test.blocks.test_utils import TestBlocks
from federatedml.util import consts


# noinspection PyUnusedLocal
def dh_call(job_id, role, ind, *args):
    if role == consts.ARBITER:
        uuid_generator.Server().validate_uuid()
        return diffie_hellman.Server().key_exchange()
    else:
        uid = uuid_generator.Client().generate_uuid()
        return uid, diffie_hellman.Client().key_exchange(uid)


class DHKeyExchangeTest(TestBlocks):

    def dh_key_exchange(self, num_hosts):
        _, guest, hosts = self.run_test(dh_call, self.job_id, num_hosts=num_hosts)
        results = [guest]
        results.extend(hosts)
        self.assertEqual(len(results), num_hosts + 1)

        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                self.assertEqual(results[i][1][results[j][0]], results[j][1][results[i][0]])

    def test_host_1(self):
        self.dh_key_exchange(1)

    def test_host_10(self):
        self.maxDiff = None
        self.dh_key_exchange(10)
