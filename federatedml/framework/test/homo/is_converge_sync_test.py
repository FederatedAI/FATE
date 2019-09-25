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

from federatedml.framework.homo.sync import is_converge_sync
from federatedml.util import consts
from .homo_test_sync_base import TestSyncBase


class IsConvergeTest(TestSyncBase):

    @classmethod
    def call(cls, role, transfer_variable, ind, *args):
        epsilon = 0.001
        funcs = [lambda x: abs(x) < epsilon, lambda x, y: abs(x - y) < epsilon]
        args = [
            [(0.00005,), (3.14159, 3.1415926)],
            [(0.0015,), (2.71, 3.14)]
        ]
        if role == consts.ARBITER:
            status = []
            is_converge = is_converge_sync.Arbiter().register_is_converge(transfer_variable.is_converge)
            for i in range(2):
                for j in range(2):
                    status.append(is_converge.check_converge_status(funcs[i], args[j][i], suffix=(i, j)))
            return status
        elif role == consts.HOST:
            status = []
            is_converge = is_converge_sync.Host().register_is_converge(transfer_variable.is_converge)
            for i in range(2):
                for j in range(2):
                    status.append(is_converge.get_converge_status(suffix=(i, j)))
            return status
        else:
            status = []
            is_converge = is_converge_sync.Guest().register_is_converge(transfer_variable.is_converge)
            for i in range(2):
                for j in range(2):
                    status.append(is_converge.get_converge_status(suffix=(i, j)))
            return status

    def run_with_num_hosts(self, num_hosts):
        arbiter, guest, *hosts = self.run_results(num_hosts=num_hosts)
        self.assertListEqual(arbiter, guest)
        for state in hosts:
            self.assertLessEqual(arbiter, state)

    def test_host_1(self):
        self.run_with_num_hosts(1)

    def test_host_10(self):
        self.run_with_num_hosts(10)
