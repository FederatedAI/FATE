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

# noinspection PyPackageRequirements
from pyspark import rddsampler, RDD, SparkContext, util

from fate_arch.common import log
from fate_arch.common.profile import log_elapsed
from fate_arch.data_table.base import AddressABC, HDFSAddress
from fate_arch.session._interface import TableABC
from fate_arch.session.impl.spark import _util
from fate_arch.session.impl.spark._kv_serdes import _load_from_hdfs, _save_as_hdfs

LOGGER = log.getLogger()


class Table(TableABC):

    def __init__(self, rdd: RDD):
        self._rdd = rdd

    @staticmethod
    def __getstate__(self):
        pass

    """unary transform
    """

    @log_elapsed
    def map(self, func, **kwargs):
        return _from_rdd(_map(self._rdd, func))

    @log_elapsed
    def mapValues(self, func, **kwargs):
        return _from_rdd(_map_value(self._rdd, func))

    @log_elapsed
    def mapPartitions(self, func, **kwargs):
        return _from_rdd(self._rdd.mapPartitions(func))

    @log_elapsed
    def glom(self, **kwargs):
        return _from_rdd(_glom(self._rdd))

    @log_elapsed
    def sample(self, fraction, seed=None, **kwargs):
        return _from_rdd(_sample(self._rdd, fraction, seed))

    @log_elapsed
    def filter(self, func, **kwargs):
        return _from_rdd(_filter(self._rdd, func))

    @log_elapsed
    def flatMap(self, func, **kwargs):
        return _from_rdd(_flat_map(self._rdd, func))

    """action
    """

    @log_elapsed
    def reduce(self, func, **kwargs):
        return self._rdd.values().reduce(func)

    @log_elapsed
    def collect(self, **kwargs):
        return iter(self._rdd.collect())

    @log_elapsed
    def take(self, n=1, **kwargs):
        return self._rdd.take(n)

    @log_elapsed
    def first(self, **kwargs):
        return self.take(1)[0]

    @log_elapsed
    def count(self, **kwargs):
        return self._rdd.count()

    @log_elapsed
    def save(self, address: AddressABC, partitions: int, schema: dict, **kwargs):
        if isinstance(address, HDFSAddress):
            _save_as_hdfs(rdd=self._rdd, paths=address.path, partitions=partitions)
            schema.update(self.schema)
        raise NotImplementedError(f"address type {type(address)} not supported with spark backend")

    """binary transform
    """

    @log_elapsed
    def join(self, other: 'Table', func=None, **kwargs):
        return _from_rdd(_join(self._rdd, other._rdd, func=func))

    @log_elapsed
    def subtractByKey(self, other: 'Table', **kwargs):
        return _from_rdd(_subtract_by_key(self._rdd, other._rdd))

    @log_elapsed
    def union(self, other: 'Table', func=lambda v1, v2: v1, **kwargs):
        return _from_rdd(_union(self._rdd, other._rdd, func))


def _from_hdfs(paths: str, partitions):
    sc = SparkContext.getOrCreate()
    rdd = _util.materialize(_load_from_hdfs(sc, paths, partitions))
    return Table(rdd=rdd)


def _from_rdd(rdd):
    rdd = _util.materialize(rdd)
    return Table(rdd=rdd)


def _map(rdd, func):
    def _fn(x):
        return func(x[0], x[1])

    def _func(_, iterator):
        return map(util.fail_on_stopiteration(_fn), iterator)

    return rdd.mapPartitionsWithIndex(_func, preservesPartitioning=False)


def _map_value(rdd, func):
    def _fn(x):
        return x[0], func(x[1])

    def _func(_, iterator):
        return map(util.fail_on_stopiteration(_fn), iterator)

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
    _sample_func = rddsampler.RDDSampler(False, fraction, seed).func

    def _func(split, iterator):
        return _sample_func(split, iterator)

    return rdd.mapPartitionsWithIndex(_func, preservesPartitioning=True)


def _filter(rdd, func):
    def _fn(x):
        return func(x[0], x[1])

    def _func(_, iterator):
        return filter(util.fail_on_stopiteration(_fn), iterator)

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
        return chain.from_iterable(map(util.fail_on_stopiteration(_fn), iterator))

    rdd.mapPartitionsWithIndex(_func, preservesPartitioning=False)
