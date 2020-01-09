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


import uuid

from pyspark import RDD
from pyspark.util import fail_on_stopiteration
from arch.api.table import eggroll_util


def _save_as_func(rdd: RDD, name, namespace, partition, persistent):
    from arch.api import session
    dup = session.table(name=name, namespace=namespace, partition=partition, persistent=persistent)

    def _func(_, it):
        eggroll_util.maybe_create_eggroll_client()
        dup.put_all(list(it))
        return 1,

    rdd.mapPartitionsWithIndex(_func, preservesPartitioning=False).collect()
    return dup


def _map(rdd: RDD, func):
    def _fn(x):
        return func(x[0], x[1])

    def _func(_, iterator):
        return map(fail_on_stopiteration(_fn), iterator)

    return rdd.mapPartitionsWithIndex(_func, preservesPartitioning=False)


def _map_value(rdd: RDD, func):
    def _fn(x):
        return x[0], func(x[1])

    def _func(_, iterator):
        return map(fail_on_stopiteration(_fn), iterator)

    return rdd.mapPartitionsWithIndex(_func, preservesPartitioning=True)


def _map_partitions(rdd: RDD, func):
    def _func(_, iterator):
        return [(str(uuid.uuid1()), func(iterator))]

    return rdd.mapPartitionsWithIndex(_func, preservesPartitioning=False)


def _join(rdd: RDD, other: RDD, func=None):
    num_partitions = max(rdd.getNumPartitions(), other.getNumPartitions())
    rtn_rdd = rdd.join(other, numPartitions=num_partitions)
    if func is not None:
        rtn_rdd = _map_value(rtn_rdd, lambda x: func(x[0], x[1]))
    return rtn_rdd


def _glom(rdd: RDD):
    def _func(_, iterator):
        yield list(iterator)

    return rdd.mapPartitionsWithIndex(_func)


def _sample(rdd: RDD, fraction: float, seed: int):
    from pyspark.rddsampler import RDDSampler
    assert fraction >= 0.0, "Negative fraction value: %s" % fraction

    _sample_func = RDDSampler(False, fraction, seed).func

    def _func(split, iterator):
        return _sample_func(split, iterator)

    return rdd.mapPartitionsWithIndex(_func, preservesPartitioning=True)


def _filter(rdd: RDD, func):
    def _fn(x):
        return func(x[0], x[1])

    def _func(_, iterator):
        return filter(fail_on_stopiteration(_fn), iterator)

    return rdd.mapPartitionsWithIndex(_func, preservesPartitioning=True)


def _subtract_by_key(rdd: RDD, other: RDD):
    return rdd.subtractByKey(other, rdd.getNumPartitions())


def _union(rdd: RDD, other: RDD, func):
    num_partition = max(rdd.getNumPartitions(), other.getNumPartitions())

    def _func(pair):
        iter1, iter2 = pair
        val1 = list(iter1)
        val2 = list(iter2)
        if not val1:
            return val2[0]
        if not val2:
            return val1[0]
        return func(val1[0], val2[0])

    return _map_value(rdd.cogroup(other, num_partition), _func)


def _flat_map(rdd: RDD, func):
    from itertools import chain

    def _fn(x):
        return func(x[0], x[1])

    def _func(_, iterator):
        return chain.from_iterable(map(fail_on_stopiteration(_fn), iterator))

    rdd.mapPartitionsWithIndex(_func, preservesPartitioning=False)
