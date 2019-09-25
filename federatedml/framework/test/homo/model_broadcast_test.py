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

from federatedml.framework.homo.sync import model_broadcast_sync
from federatedml.util import consts
from .homo_test_sync_base import TestSyncBase


class ModelBroadcastTest(TestSyncBase):

    @classmethod
    def call(cls, role, transfer_variable, ind, *args):
        cipher_dict = args[0]
        vars_to_broadcast = args[1]
        if role == consts.ARBITER:
            return model_broadcast_sync.Arbiter() \
                .register_model_broadcaster(transfer_variable.aggregated_model) \
                .send_model(vars_to_broadcast, cipher_dict)
        elif role == consts.HOST:
            return model_broadcast_sync.Host() \
                .register_model_broadcaster(transfer_variable.aggregated_model) \
                .get_model()
        else:
            return model_broadcast_sync.Guest() \
                .register_model_broadcaster(transfer_variable.aggregated_model) \
                .get_model()

    def run_with_num_hosts(self, num_hosts):
        ratio = 0.3
        key_size = 1024

        import random
        from federatedml.secureprotol.encrypt import PaillierEncrypt
        PaillierEncrypt().generate_key(key_size)
        cipher_dict = {}
        for i in range(num_hosts):
            if random.random() > ratio:
                cipher = PaillierEncrypt()
                cipher.generate_key(key_size)
                cipher_dict[i] = cipher
            else:
                cipher_dict[i] = None

        from federatedml.framework.weights import ListWeights
        variables = ListWeights([random.random() for _ in range(10)])

        arbiter, guest, *hosts = self.run_results(num_hosts, cipher_dict, variables)
        guest = guest.unboxed
        hosts = [host.unboxed for host in hosts]
        self.assertListEqual(guest, variables.unboxed)

        host_decrypted = [cipher_dict[i].decrypt_list(hosts[i]) if cipher_dict[i] else hosts[i]
                          for i in range(num_hosts)]
        for i in range(len(guest)):
            for j in range(num_hosts):
                self.assertAlmostEqual(guest[i], host_decrypted[j][i])

    def test_host_1(self):
        self.run_with_num_hosts(1)

    def test_host_10(self):
        self.run_with_num_hosts(10)
