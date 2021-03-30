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

from federatedml.framework.weights import ListWeights
from federatedml.util import consts
from .homo_test_sync_base import TestSyncBase
from federatedml.framework.homo.procedure import paillier_cipher


class PaillierCipherTest(TestSyncBase):
    @classmethod
    def call(cls, role, transfer_variable, ind, *args):
        iter_num = 7
        re_encrypt_batches = 3
        encrypt_rate = 0.3
        key_size = 1024

        if role == consts.ARBITER:
            agg = paillier_cipher.Arbiter()
            agg.register_paillier_cipher(transfer_variable)
            cipher_dict = agg.paillier_keygen(key_size)
            re_cipher_time = agg.set_re_cipher_time(cipher_dict)
            agg.re_cipher(iter_num, re_cipher_time, cipher_dict, re_encrypt_batches)
            return re_cipher_time, cipher_dict

        elif role == consts.HOST:
            import random
            enable = random.random() > encrypt_rate
            agg = paillier_cipher.Host()
            agg.register_paillier_cipher(transfer_variable)
            host_cipher = agg.gen_paillier_pubkey(enable)
            if enable:
                re_cipher_time = random.randint(1, 5)
                agg.set_re_cipher_time(re_encrypt_times=re_cipher_time)
                init_w = [random.random()]
                w = [host_cipher.encrypt(v) for v in init_w]
                for i in range(re_cipher_time):
                    w = agg.re_cipher(w, iter_num, i * re_encrypt_batches)
                return re_cipher_time, init_w, w

        else:
            pass

    def run_with_num_hosts(self, num_hosts):
        arbiter, guest, *hosts = self.run_results(num_hosts=num_hosts)
        for i in range(len(hosts)):
            if hosts[i] is not None:
                self.assertEqual(hosts[i][0], arbiter[0][i])
                final_decrypted = arbiter[1][i].decrypt_list(hosts[i][2])
                init_w = hosts[i][1]
                for j in range(len(final_decrypted)):
                    self.assertAlmostEqual(init_w[j], final_decrypted[j])

    def test_host_1(self):
        self.run_with_num_hosts(1)

    def test_host_10(self):
        self.run_with_num_hosts(10)
