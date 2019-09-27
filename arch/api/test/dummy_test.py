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

from arch.api.session import init, parallelize
# from arch.api.cluster.mock_roll import init, parallelize

import numpy as np


def f(iterator):
    sum = 0
    for k, v in iterator:
        sum += v
    return sum


if __name__ == "__main__":
    init()

    _matrix = np.ones([400, 50])

    _table = parallelize(_matrix, partition=40)

    c = _table.mapValues(lambda _x: _x)
    dict(c.collect())
    print(list(c.collect()))

    _table = parallelize(["b", "a", "c"], partition=5)

    a = _table.mapValues(lambda _x: _x + "1")
    print(list(a.collect()))
    print(dict(a.collect()))
    print(list(_table.collect()))
    x = _table.map(lambda k, v: (v, v + "1"))
    print(list(x.collect()))
    _table = parallelize([("a", ["apple", "banana", "lemon"]), ("b", ["grapes"])], include_key=True)
    print(list(_table.mapValues(lambda _x: len(_x)).collect()))
    _table = parallelize([1, 2, 3, 4, 5], partition=2)
    print(list(_table.mapPartitions(f).collect()))
    from operator import add

    print(parallelize([1, 2, 3, 4, 5], partition=4).reduce(add))
    x = parallelize([("a", 1), ("b", 4)], include_key=True)
    y = parallelize([("a", 2), ("c", 3)], include_key=True)
    print(list(x.join(y, lambda v1, v2: v1 + v2).collect()))
    x = parallelize(range(100), partition=4)
    print(x.sample(0.1, 81).count())
    print(list(parallelize([0, 2, 3, 4, 6], partition=5).glom().collect()))
