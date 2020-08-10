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
import random
import unittest
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from federatedml.secureprotol.spdz import SPDZ
from federatedml.secureprotol.spdz.tensor.fixedpoint_numpy import FixedPointTensor
from federatedml.transfer_variable.transfer_class.secret_share_transfer_variable import SecretShareTransferVariable

NUM_HOSTS = 1
EPS = 0.001


def session_init(job_id, idx):
    from arch.api import session
    from arch.api import federation

    role = "guest" if idx < 1 else "host"
    party_id = 9999 + idx if idx < 1 else 10000 + (idx - 1)
    role_parties = {
        "host": [
            10000 + i for i in range(NUM_HOSTS)
        ],
        "guest": [
            9999 + i for i in range(1)
        ]
    }
    session.init(job_id)
    federation.init(job_id, dict(local=dict(role=role, party_id=party_id),
                                 role=role_parties))
    return federation.local_party(), federation.all_parties()


def submit(func, *args, **kwargs):
    with ProcessPoolExecutor() as pool:
        num = NUM_HOSTS + 1
        result = [None] * num
        futures = {}
        for _idx in range(num):
            kv = kwargs.copy()
            kv["idx"] = _idx
            futures[pool.submit(func, *args, **kv)] = _idx
        for future in as_completed(futures):
            result[futures[future]] = future.result()
        return result


def create_and_get(job_id, idx, data):
    _, all_parties = session_init(job_id, idx)
    with SPDZ():
        if idx == 0:
            x = FixedPointTensor.from_source("x", data)
        else:
            x = FixedPointTensor.from_source("x", all_parties[0])
        return x.get()


def add_and_sub(job_id, idx, data_list):
    _, all_parties = session_init(job_id, idx)
    with SPDZ():
        if idx == 0:
            x = FixedPointTensor.from_source("x", data_list[0])
            y = FixedPointTensor.from_source("y", all_parties[1])
        else:
            x = FixedPointTensor.from_source("x", all_parties[0])
            y = FixedPointTensor.from_source("y", data_list[1])
        a = (x + y).get()
        b = (x - y).get()
        return a, b


def add_and_sub_plaintext(job_id, idx, data_list):
    _, all_parties = session_init(job_id, idx)
    with SPDZ():
        if idx == 0:
            x = FixedPointTensor.from_source("x", data_list[0])
        else:
            x = FixedPointTensor.from_source("x", all_parties[0])
        y = data_list[1]
        a = (x + y).get()
        a1 = (y + x).get()
        b = (x - y).get()
        b1 = (y - x).get()
        return a, a1, b, b1


def mul_plaintext(job_id, idx, data_list):
    _, all_parties = session_init(job_id, idx)
    with SPDZ():
        if idx == 0:
            x = FixedPointTensor.from_source("x", data_list[0])
        else:
            x = FixedPointTensor.from_source("x", all_parties[0])
        y = data_list[1]
        return (x * y).get(), (y * x).get()


def mat_mul(job_id, idx, data_list):
    _, all_parties = session_init(job_id, idx)
    with SPDZ():
        if idx == 0:
            x = FixedPointTensor.from_source("x", data_list[0])
            y = FixedPointTensor.from_source("y", all_parties[1])
        else:
            x = FixedPointTensor.from_source("x", all_parties[0])
            y = FixedPointTensor.from_source("y", data_list[1])
        return (x @ y).get()


def einsum(job_id, idx, einsum_expr, data_list):
    _, all_parties = session_init(job_id, idx)
    with SPDZ():
        if idx == 0:
            x = FixedPointTensor.from_source("x", data_list[0])
            y = FixedPointTensor.from_source("y", all_parties[1])
        else:
            x = FixedPointTensor.from_source("x", all_parties[0])
            y = FixedPointTensor.from_source("y", data_list[1])
        return x.einsum(y, einsum_expr).get()


class TestSyncBase(unittest.TestCase):

    def setUp(self) -> None:
        self.transfer_variable = SecretShareTransferVariable()
        self.job_id = str(uuid.uuid1())
        self.transfer_variable.set_flowid(self.job_id)

    def test_create_and_get(self):
        data = np.random.rand(10, 15)
        rec = submit(create_and_get, self.job_id, data=data)
        for x in rec:
            self.assertAlmostEqual(np.linalg.norm(x - data), 0, delta=EPS)

    def test_add_and_sub(self):
        x = np.random.rand(10, 15)
        y = np.random.rand(10, 15)
        data_list = [x, y]
        rec = submit(add_and_sub, self.job_id, data_list=data_list)
        for a, b in rec:
            self.assertAlmostEqual(np.linalg.norm((x + y) - a), 0, delta=2 * EPS)
            self.assertAlmostEqual(np.linalg.norm((x - y) - b), 0, delta=2 * EPS)

    def test_add_and_sub_plaintext(self):
        # x = np.random.rand(10, 15)
        # y = np.random.rand(10, 15)
        x = np.array([1, 2, 3, 4])
        y = np.array([5, 6, 7, 8])
        data_list = [x, y]
        rec = submit(add_and_sub_plaintext, self.job_id, data_list=data_list)
        for a, a1, b, b1 in rec:
            self.assertAlmostEqual(np.linalg.norm((x + y) - a), 0, delta=2 * EPS)
            self.assertAlmostEqual(np.linalg.norm((x + y) - a1), 0, delta=2 * EPS)
            self.assertAlmostEquals(np.linalg.norm((x - y) - b), 0, delta=2 * EPS)
            self.assertAlmostEquals(np.linalg.norm((y - x) - b1), 0, delta=2 * EPS)

    def test_mul_plaintext(self):
        x = np.random.rand(10, 15)
        y = random.randint(1, 10000)
        data_list = [x, y]
        rec = submit(mul_plaintext, self.job_id, data_list=data_list)
        for a, b in rec:
            self.assertAlmostEqual(np.linalg.norm((x * y) - a), 0, delta=y * EPS)
            self.assertAlmostEqual(np.linalg.norm((x * y) - b), 0, delta=y * EPS)

    def test_matmul(self):
        j_dim = 15
        x = np.random.rand(10, j_dim)
        y = np.random.rand(j_dim, 20)
        data_list = [x, y]
        rec = submit(mat_mul, self.job_id, data_list=data_list)
        for a in rec:
            self.assertAlmostEqual(np.linalg.norm((x @ y) - a), 0, delta=j_dim * EPS)

    def test_einsum(self):
        j_dim = 5
        k_dim = 4
        x = np.random.rand(10, j_dim, k_dim)
        y = np.random.rand(k_dim, j_dim, 20)
        einsum_expr = "ijk,kjl->il"
        data_list = [x, y]
        rec = submit(einsum, self.job_id, einsum_expr=einsum_expr, data_list=data_list)
        for a in rec:
            self.assertAlmostEqual(np.linalg.norm(np.einsum(einsum_expr, x, y) - a), 0, delta=j_dim * k_dim * EPS)
