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
from itertools import chain

from fate_arch.abc import CTableABC
from fate_arch.common import log, hdfs_utils
from fate_arch.common.profile import computing_profile
from fate_arch.computing.spark._materialize import materialize

LOGGER = log.getLogger()


class Table(CTableABC):

    def __init__(self, rdd):
        self._rdd = rdd

    def __getstate__(self):
        pass

    @computing_profile
    def save(self, address, partitions, schema, **kwargs):
        from fate_arch.common.address import HDFSAddress
        if isinstance(address, HDFSAddress):
            self._rdd.map(lambda x: hdfs_utils.serialize(x[0], x[1])) \
                .repartition(partitions) \
                .saveAsTextFile(address.path)
            schema.update(self.schema)
            return
        raise NotImplementedError(f"address type {type(address)} not supported with spark backend")

    @property
    def partitions(self):
        return self._rdd.getNumPartitions()

    @computing_profile
    def map(self, func, **kwargs):
        return from_rdd(_map(self._rdd, func))

    @computing_profile
    def mapValues(self, func, **kwargs):
        return from_rdd(_map_value(self._rdd, func))

    @computing_profile
    def mapPartitions(self, func, **kwargs):
        return from_rdd(self._rdd.mapPartitions(func))

    @computing_profile
    def mapReducePartitions(self, mapper, reducer, **kwargs):
        return from_rdd(self._rdd.mapPartitions(mapper).reduceByKey(reducer))

    @computing_profile
    def applyPartitions(self, func, use_previous_behavior=True, **kwargs):
        if use_previous_behavior is True:
            LOGGER.warning(f"please use `applyPartitions` instead of `mapPartitions` "
                           f"if the previous behavior was expected. "
                           f"The previous behavior will not work in future")
            return self.applyPartitions(func)
        return from_rdd(_map_partitions(self._rdd, func))

    @computing_profile
    def glom(self, **kwargs):
        return from_rdd(_glom(self._rdd))

    @computing_profile
    def sample(self, fraction, seed=None, **kwargs):
        return from_rdd(_sample(self._rdd, fraction, seed))

    @computing_profile
    def filter(self, func, **kwargs):
        return from_rdd(_filter(self._rdd, func))

    @computing_profile
    def flatMap(self, func, **kwargs):
        return from_rdd(_flat_map(self._rdd, func))

    @computing_profile
    def reduce(self, func, **kwargs):
        return self._rdd.values().reduce(func)

    @computing_profile
    def collect(self, **kwargs):
        return iter(self._rdd.collect())

    @computing_profile
    def take(self, n=1, **kwargs):
        return self._rdd.take(n)

    @computing_profile
    def first(self, **kwargs):
        return self.take(1)[0]

    @computing_profile
    def count(self, **kwargs):
        return self._rdd.count()

    @computing_profile
    def join(self, other: 'Table', func=None, **kwargs):
        return from_rdd(_join(self._rdd, other._rdd, func=func))

    @computing_profile
    def subtractByKey(self, other: 'Table', **kwargs):
        return from_rdd(_subtract_by_key(self._rdd, other._rdd))

    @computing_profile
    def union(self, other: 'Table', func=lambda v1, v2: v1, **kwargs):
        return from_rdd(_union(self._rdd, other._rdd, func))


def from_hdfs(paths: str, partitions):
    # noinspection PyPackageRequirements
    from pyspark import SparkContext
    sc = SparkContext.getOrCreate()
    rdd = materialize(sc.textFile(paths, partitions).map(hdfs_utils.deserialize).repartition(partitions))
    return Table(rdd=rdd)


def from_rdd(rdd):
    rdd = materialize(rdd)
    return Table(rdd=rdd)


def _fail_on_stopiteration(fn):
    # noinspection PyPackageRequirements
    from pyspark import util
    return util.fail_on_stopiteration(fn)


def _map(rdd, func):
    def _fn(x):
        return func(x[0], x[1])

    def _func(_, iterator):
        return map(_fail_on_stopiteration(_fn), iterator)

    return rdd.mapPartitionsWithIndex(_func, preservesPartitioning=False)


def _map_value(rdd, func):
    def _fn(x):
        return x[0], func(x[1])

    def _func(_, iterator):
        return map(_fail_on_stopiteration(_fn), iterator)

    return rdd.mapPartitionsWithIndex(_func, preservesPartitioning=True)


def _map_partitions(rdd, func):
    def _func(_, iterator):
        return [(str(uuid.uuid1()), func(iterator))]

    return rdd.mapPartitionsWithIndex(_func, preservesPartitioning=False)


def _join(rdd, other, func=None):
    num_partitions = max(rdd.getNumPartitions(), other.getNumPartitions())
    rtn_rdd = rdd.join(other, numPartitions=num_partitions)
    if func is not None:
        rtn_rdd = _map_value(rtn_rdd, lambda x: func(x[0], x[1]))
    return rtn_rdd


def _glom(rdd):
    def _func(_, iterator):
        yield list(iterator)

    return rdd.mapPartitionsWithIndex(_func)


def _sample(rdd, fraction: float, seed: int):
    assert fraction >= 0.0, "Negative fraction value: %s" % fraction
    # noinspection PyPackageRequirements
    from pyspark import rddsampler
    _sample_func = rddsampler.RDDSampler(False, fraction, seed).func

    def _func(split, iterator):
        return _sample_func(split, iterator)

    return rdd.mapPartitionsWithIndex(_func, preservesPartitioning=True)


def _filter(rdd, func):
    def _fn(x):
        return func(x[0], x[1])

    def _func(_, iterator):
        return filter(_fail_on_stopiteration(_fn), iterator)

    return rdd.mapPartitionsWithIndex(_func, preservesPartitioning=True)


def _subtract_by_key(rdd, other):
    return rdd.subtractByKey(other, rdd.getNumPartitions())


def _union(rdd, other, func):
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


def _flat_map(rdd, func):
    def _fn(x):
        return func(x[0], x[1])

    def _func(_, iterator):
        return chain.from_iterable(map(_fail_on_stopiteration(_fn), iterator))

    rdd.mapPartitionsWithIndex(_func, preservesPartitioning=False)
