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

from federatedml.framework.homo.sync import paillier_re_cipher_sync, paillier_keygen_sync
from federatedml.util import consts
from .homo_test_sync_base import TestSyncBase


class PaillierReCipherTest(TestSyncBase):

    @classmethod
    def call(cls, role, transfer_variable, ind, *args):
        iter_num = 4
        re_encrypt_batches = 2
        encrypt_rate = 0.3
        key_size = 1024

        if role == consts.ARBITER:
            host_ciphers = paillier_keygen_sync.Arbiter() \
                ._register_paillier_keygen(transfer_variable.use_encrypt,
                                           transfer_variable.paillier_pubkey) \
                .paillier_keygen(key_size)
            re_cipher = paillier_re_cipher_sync.Arbiter() \
                ._register_paillier_re_cipher(transfer_variable.re_encrypt_times,
                                              transfer_variable.model_to_re_encrypt,
                                              transfer_variable.model_re_encrypted)
            re_cipher_time = re_cipher.set_re_cipher_time(host_ciphers_dict=host_ciphers)
            re_cipher.re_cipher(iter_num, re_cipher_time, host_ciphers, re_encrypt_batches)
            return re_cipher_time, host_ciphers
        elif role == consts.HOST:
            import random
            enable = random.random() > encrypt_rate
            host_cipher = paillier_keygen_sync.Host() \
                ._register_paillier_keygen(transfer_variable.use_encrypt,
                                           transfer_variable.paillier_pubkey) \
                .gen_paillier_pubkey(enable=enable)
            if enable:
                re_cipher = paillier_re_cipher_sync.Host() \
                    ._register_paillier_re_cipher(transfer_variable.re_encrypt_times,
                                                  transfer_variable.model_to_re_encrypt,
                                                  transfer_variable.model_re_encrypted)
                re_cipher_time = random.randint(1, 5)
                re_cipher.set_re_cipher_time(re_cipher_time)

                init_w = [random.random()]
                w = [host_cipher.encrypt(v) for v in init_w]
                for i in range(re_cipher_time):
                    w = re_cipher.re_cipher(w, iter_num, (i+1) * re_encrypt_batches)
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
