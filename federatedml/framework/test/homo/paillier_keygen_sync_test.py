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

from federatedml.framework.homo.sync import paillier_keygen_sync
from federatedml.util import consts
from .homo_test_sync_base import TestSyncBase


class PaillierKeyGenTest(TestSyncBase):

    @classmethod
    def call(cls, role, transfer_variable, ind, *args):
        if role == consts.ARBITER:
            return paillier_keygen_sync.Arbiter() \
                ._register_paillier_keygen(transfer_variable.use_encrypt,
                                           transfer_variable.paillier_pubkey) \
                .paillier_keygen(1024)
        elif role == consts.HOST:
            import random
            enable = random.random() > 0.3
            return paillier_keygen_sync.Host() \
                ._register_paillier_keygen(transfer_variable.use_encrypt,
                                           transfer_variable.paillier_pubkey)\
                .gen_paillier_pubkey(enable=enable)
        else:
            pass

    def run_with_num_hosts(self, num_hosts):
        arbiter, guest, *hosts = self.run_results(num_hosts=num_hosts)
        enabled = []
        for i in range(len(hosts)):
            if hosts[i]:
                enabled.append(i)
        self.assertEqual(len([x for x in arbiter.values() if x]), len(enabled))
        for i in enabled:
            self.assertEqual(arbiter[i].get_public_key(), hosts[i])

    def test_host_1(self):
        self.run_with_num_hosts(1)

    def test_host_10(self):
        self.run_with_num_hosts(10)
