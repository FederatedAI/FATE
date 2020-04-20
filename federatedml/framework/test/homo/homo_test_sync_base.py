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
import unittest
import uuid
from multiprocessing import Pool

from federatedml.util import consts
from federatedml.transfer_variable.transfer_class.homo_transfer_variable import HomoTransferVariable
import time


class TestSyncBase(unittest.TestCase):

    @classmethod
    def init_table_manager_and_federation(cls, job_id, role, num_hosts, host_ind=0):
        from arch.api import session
        from arch.api import federation

        role_id = {
            "host": [
                10000 + i for i in range(num_hosts)
            ],
            "guest": [
                9999
            ],
            "arbiter": [
                9999
            ]
        }
        session.init(job_id)
        federation.init(job_id,
                        {"local": {
                            "role": role,
                            "party_id": role_id[role][0] if role != "host" else role_id[role][host_ind]
                        },
                            "role": role_id
                        })

    def clean_tables(self):
        from arch.api import session
        session.init(job_id=self.job_id)
        try:
            session.cleanup("*", self.job_id, True)
        except EnvironmentError:
            pass
        try:
            session.cleanup("*", self.job_id, False)
        except EnvironmentError:
            pass

    def setUp(self) -> None:
        self.transfer_variable = HomoTransferVariable()
        self.job_id = str(uuid.uuid1())
        self.transfer_variable.set_flowid(self.job_id)

    def tearDown(self) -> None:
        pass
        # self.clean_tables()

    @classmethod
    def _call(cls, job_id, role, transfer_variable, num_hosts, ind, *args):
        cls.init_table_manager_and_federation(job_id, role, num_hosts, ind)
        return cls.call(role, transfer_variable, ind, *args)

    @classmethod
    def call(cls, role, transfer_variable, ind, *args):
        pass

    @classmethod
    def results(cls, job_id, transfer_variable, num_hosts, *args):
        tasks = []
        with Pool(num_hosts + 2) as p:
            tasks.append(p.apply_async(func=cls._call,
                                        args=(job_id, consts.ARBITER, transfer_variable, num_hosts, 0, *args)))
            tasks.append(p.apply_async(func=cls._call,
                                        args=(job_id, consts.GUEST, transfer_variable, num_hosts, 0, *args)))
            for i in range(num_hosts):
                tasks.append(
                    p.apply_async(func=cls._call,
                                  args=(job_id, consts.HOST, transfer_variable, num_hosts, i, *args)))

            left = list(range(len(tasks)))
            while len(left) > 0:
                time.sleep(0.1)
                tmp = []
                for i in left:
                    if tasks[i].ready():
                        tasks[i] = tasks[i].get()
                    else:
                        tmp.append(i)
                left = tmp

            return tasks

    def run_results(self, num_hosts, *args):
        return self.results(self.job_id, self.transfer_variable, num_hosts, *args)
