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

from federatedml.framework.homo.sync import identify_uuid_sync, dh_keys_exchange_sync
from federatedml.util import consts
from .homo_test_sync_base import TestSyncBase


class DHKeyExchangeTest(TestSyncBase):

    @classmethod
    def call(cls, role, transfer_variable, ind, *args):
        if role == consts.ARBITER:
            identify_uuid_sync.Arbiter() \
                .register_identify_uuid(transfer_variable.guest_uuid,
                                        transfer_variable.host_uuid,
                                        transfer_variable.uuid_conflict_flag) \
                .validate_uuid()
            return dh_keys_exchange_sync.Arbiter() \
                .register_dh_key_exchange(transfer_variable.dh_pubkey,
                                          transfer_variable.dh_ciphertext_host,
                                          transfer_variable.dh_ciphertext_guest,
                                          transfer_variable.dh_ciphertext_bc) \
                .key_exchange()
        elif role == consts.HOST:
            uid = identify_uuid_sync.Host() \
                .register_identify_uuid(transfer_variable.host_uuid,
                                         conflict_flag_transfer_variable=transfer_variable.uuid_conflict_flag) \
                .generate_uuid()
            return (uid,
                    dh_keys_exchange_sync.Host()
                    .register_dh_key_exchange(transfer_variable.dh_pubkey,
                                               transfer_variable.dh_ciphertext_host,
                                               transfer_variable.dh_ciphertext_bc)
                    .key_exchange(uid))
        else:
            uid = identify_uuid_sync.Guest() \
                .register_identify_uuid(transfer_variable.guest_uuid,
                                         conflict_flag_transfer_variable=transfer_variable.uuid_conflict_flag) \
                .generate_uuid()
            return (uid,
                    dh_keys_exchange_sync.Guest()
                    .register_dh_key_exchange(transfer_variable.dh_pubkey,
                                               transfer_variable.dh_ciphertext_guest,
                                               transfer_variable.dh_ciphertext_bc)
                    .key_exchange(uid))

    def dh_key_exchange(self, num_hosts):
        results = self.run_results(num_hosts=num_hosts)
        self.assertEqual(len(results), num_hosts + 2)

        for i in range(1, len(results)):
            for j in range(i + 1, len(results)):
                self.assertEqual(results[i][1][results[j][0]], results[j][1][results[i][0]])

    def test_host_1(self):
        self.dh_key_exchange(1)

    def test_host_10(self):
        self.maxDiff = None
        self.dh_key_exchange(10)
