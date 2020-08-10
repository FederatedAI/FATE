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

from federatedml.framework.homo.blocks import random_padding_cipher
from federatedml.framework.homo.test.blocks.test_utils import TestBlocks
from federatedml.util import consts


# noinspection PyUnusedLocal
def sync_random_padding(job_id, role, ind, *args):
    if role == consts.ARBITER:
        rp_cipher = random_padding_cipher.Server()
        rp_cipher.exchange_secret_keys()
        return

    elif role == consts.HOST:
        rp_cipher = random_padding_cipher.Client()
        rp_cipher.create_cipher()
        return rp_cipher
    else:
        rp_cipher = random_padding_cipher.Client()
        rp_cipher.create_cipher()
        return rp_cipher


class RandomPaddingCipherTest(TestBlocks):

    def run_with_num_hosts(self, num_hosts):
        _, guest, hosts = self.run_test(sync_random_padding, self.job_id, num_hosts=num_hosts)
        import numpy as np
        raw = np.zeros((10, 10))
        encrypted = np.zeros((10, 10))

        guest_matrix = np.random.rand(10, 10)
        raw += guest_matrix
        encrypted += guest.encrypt(guest_matrix)

        for host in hosts:
            host_matrix = np.random.rand(10, 10)
            raw += host_matrix
            encrypted += host.encrypt(host_matrix)

        self.assertTrue(np.allclose(raw, encrypted))

    def test_host_1(self):
        self.run_with_num_hosts(1)

    def test_host_10(self):
        self.run_with_num_hosts(10)
