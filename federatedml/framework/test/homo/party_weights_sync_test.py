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

from federatedml.framework.homo.sync import party_weights_sync
from federatedml.util import consts
from .homo_test_sync_base import TestSyncBase


class PartyWeightsTest(TestSyncBase):

    @classmethod
    def call(cls, role, transfer_variable, ind, *args):
        if role == consts.ARBITER:
            return party_weights_sync.Arbiter() \
                ._register_party_weights_transfer(transfer_variable.guest_party_weight,
                                                  transfer_variable.host_party_weight) \
                .get_party_weights()
        elif role == consts.HOST:
            import random
            return party_weights_sync.Host() \
                ._register_party_weights_transfer(transfer_variable.host_party_weight) \
                .send_party_weight(random.random())
        else:
            import random
            return party_weights_sync.Guest() \
                ._register_party_weights_transfer(transfer_variable.guest_party_weight) \
                .send_party_weight(random.random())

    def run_with_num_hosts(self, num_hosts):
        arbiter, guest, *hosts = self.run_results(num_hosts=num_hosts)
        total = guest + sum(hosts)
        self.assertAlmostEqual(arbiter[0], guest / total)
        for i in range(1, len(arbiter)):
            self.assertAlmostEqual(arbiter[i], hosts[i-1] / total)

    def test_host_1(self):
        self.run_with_num_hosts(1)

    def test_host_10(self):
        self.run_with_num_hosts(10)
