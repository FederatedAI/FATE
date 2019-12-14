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
import time
import uuid
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from arch.api import session, federation
from arch.api.transfer import Party
from federatedml.secureprotol.spdz import SPDZ
from federatedml.secureprotol.spdz.tensor.numpy_fix_point import FixPointTensor

Q_BITS = 60
Q_FIELD = 2 << Q_BITS

num_hosts = 1
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
# p3 = Party("host", 10001)
shape_a = (2, 3)
shape_b = (3, 2)
einsum_expr = "ij,ik->jk"
parties = [p1, p2]

data = [0.5 - np.random.rand(569, 10),
        0.5 - np.random.rand(569, 20)]


# session.init(job_id)
# xx = session.parallelize(enumerate(data[0]), include_key=True)
# yy = session.parallelize(enumerate(data[1]), include_key=True)


def tensor_source(name, idx):
    d = {'x': (0, data[0]),
         'y': (1, data[1])}
    return parties[d[name][0]] if d[name][0] != idx else d[name][1]


def call(idx):
    init(idx)

    with SPDZ(q_field=Q_FIELD) as spdz:
        if idx == 0:
            x = FixPointTensor.from_source("x", data[0])
            y = FixPointTensor.from_source("y", parties[1])
        else:
            x = FixPointTensor.from_source("x", parties[0])
            y = FixPointTensor.from_source("y", data[1])
        ret = x.einsum(y, einsum_expr)
        return ret.get()


func = call
pool = ProcessPoolExecutor()

futures = []
for _idx in range(len(parties)):
    futures.append(pool.submit(func, _idx))

while True:
    if all([r.done() for r in futures]):
        break
    time.sleep(0.1)

results = [r.result() for r in futures]
calc = results[0]
right = np.einsum(einsum_expr, data[0], data[1], optimize=True)
print(np.max(np.abs(calc - right)))
