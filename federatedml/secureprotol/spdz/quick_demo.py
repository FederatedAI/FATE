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
import pprint
import time
import uuid
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from arch.api import session, federation
from arch.api.transfer import Party
from federatedml.secureprotol.spdz.tensor import SPDZ

num_hosts = 2
num_guest = 1
job_id = str(uuid.uuid1())


def init(idx):
    role = "guest" if idx < num_guest else "host"
    party_id = 9999 + idx if idx < num_guest else 10000 + (idx - num_guest)
    role_parties = {
        "host": [
            10000 + i for i in range(num_hosts)
        ],
        "guest": [
            9999 + i for i in range(num_guest)
        ]
    }
    session.init(job_id)
    federation.init(job_id, dict(local=dict(role=role, party_id=party_id), role=role_parties))


p1 = Party("guest", 9999)
p2 = Party("host", 10000)
p3 = Party("host", 10001)
shape_a = (2, 3)
shape_b = (3, 2)
einsum_expr = "ij,jk->ik"
parties = [p1, p2, p3]

xx = np.array([[1, 2], [3, 4]])
yy = np.array([[2, 3], [4, 5]])
zz = np.array([[3, 4, 5], [5, 6, 7]])


def tensor_source(name, idx):
    d = {'x': (0, xx),
         'y': (1, yy),
         'z': (2, zz)}
    return parties[d[name][0]] if d[name][0] != idx else d[name][1]


def call(idx):
    init(idx)

    with SPDZ("a name") as spdz:
        x = spdz.share("x", tensor_source("x", idx))
        y = spdz.share("y", tensor_source("y", idx))
        z = spdz.share("z", tensor_source("z", idx))

        return (x @ y @ z).rescontruct()


pool = ProcessPoolExecutor()

futures = []
for _idx in range(len(parties)):
    futures.append(pool.submit(call, _idx))

while True:
    if all([r.done() for r in futures]):
        break
    time.sleep(0.1)

results = [r.result() for r in futures]
pprint.pprint(results)
pprint.pprint(xx @ yy @ zz)
