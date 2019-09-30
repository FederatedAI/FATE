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
from federatedml.framework.homo.sync import identify_uuid_sync
from federatedml.util import consts
from .homo_test_sync_base import TestSyncBase


class IdentifyUUIDTest(TestSyncBase):

    @classmethod
    def call(cls, role, transfer_variable, ind, *args):
        if role == consts.ARBITER:
            sync = identify_uuid_sync.Arbiter()
            sync.register_identify_uuid(transfer_variable.guest_uuid,
                                        transfer_variable.host_uuid,
                                        transfer_variable.uuid_conflict_flag)
            return sync.validate_uuid()
        elif role == consts.HOST:
            sync = identify_uuid_sync.Host()
            return sync.register_identify_uuid(transfer_variable.host_uuid,
                                               conflict_flag_transfer_variable=transfer_variable.uuid_conflict_flag) \
                .generate_uuid()
        else:
            sync = identify_uuid_sync.Guest()
            sync.register_identify_uuid(transfer_variable.guest_uuid,
                                        conflict_flag_transfer_variable=transfer_variable.uuid_conflict_flag)
            return sync.generate_uuid()

    def run_with_num_hosts(self, num_hosts):
        results = self.run_results(num_hosts=num_hosts)
        arbiter = results[0]
        guest = results[1]
        hosts = results[2:]

        self.assertEqual(len(results), num_hosts + 2)
        self.assertIn(guest, arbiter)
        for host in hosts:
            self.assertIn(host, arbiter)

    def test_host_1(self):
        self.run_with_num_hosts(1)

    def test_host_10(self):
        self.run_with_num_hosts(10)
