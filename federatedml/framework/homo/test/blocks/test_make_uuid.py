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
from federatedml.framework.homo.blocks import uuid_generator
from federatedml.util import consts
from federatedml.framework.homo.test.blocks.test_utils import TestBlocks


# noinspection PyProtectedMember,PyUnusedLocal
def uuid_call(job_id, role, ind, *args):
    if role == consts.ARBITER:
        uuid_server = uuid_generator.Server()
        uuid_server.validate_uuid()
        return uuid_server._uuid_set
    else:
        uuid_client = uuid_generator.Client()
        uid = uuid_client.generate_uuid()
        return uid


class IdentifyUUIDTest(TestBlocks):

    def run_uuid_test(self, num_hosts):
        uuid_set, guest_uuid, hosts_uuid = self.run_test(uuid_call, self.job_id, num_hosts=num_hosts)
        self.assertEqual(len(hosts_uuid), num_hosts)
        self.assertIn(guest_uuid, uuid_set)
        for host_uuid in hosts_uuid:
            self.assertIn(host_uuid, uuid_set)

    def test_host_1(self):
        self.run_uuid_test(1)

    def test_host_10(self):
        self.run_uuid_test(10)
