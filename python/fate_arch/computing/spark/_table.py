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

import typing
import pyspark

from pyspark.rddsampler import RDDSamplerBase

from fate_arch.abc import CTableABC
from fate_arch.common import log, hdfs_utils, hive_utils
from fate_arch.common.profile import computing_profile
from fate_arch.computing.spark._materialize import materialize, unmaterialize
from scipy.stats import hypergeom
from fate_arch.computing._type import ComputingEngine

LOGGER = log.getLogger()


class Table(CTableABC):
    def __init__(self, rdd):
        self._rdd: pyspark.RDD = rdd
        self._engine = ComputingEngine.SPARK

        self._count = None

    @property
    def engine(self):
        return self._engine

    def __getstate__(self):
        pass

    def __del__(self):
        try:
            unmaterialize(self._rdd)
            del self._rdd
        except BaseException:
            return

    def copy(self):
        """rdd is immutable, yet, inside content could be modify in some case"""
        return Table(_map_value(self._rdd, lambda x: x))

    @computing_profile
    def save(self, address, partitions, schema, **kwargs):
        from fate_arch.common.address import HDFSAddress

        if isinstance(address, HDFSAddress):
            self._rdd.map(lambda x: hdfs_utils.serialize(x[0], x[1])).repartition(
                partitions
            ).saveAsTextFile(f"{address.name_node}/{address.path}")
            schema.update(self.schema)
            return

        from fate_arch.common.address import HiveAddress, LinkisHiveAddress

        if isinstance(address, (HiveAddress, LinkisHiveAddress)):
            # df = (
            #     self._rdd.map(lambda x: hive_utils.to_row(x[0], x[1]))
            #     .repartition(partitions)
            #     .toDF()
            # )
            LOGGER.debug(f"partitions: {partitions}")
            _repartition = self._rdd.map(lambda x: hive_utils.to_row(x[0], x[1])).repartition(partitions)
            _repartition.toDF().write.saveAsTable(f"{address.database}.{address.name}")
            schema.update(self.schema)
            return

        from fate_arch.common.address import LocalFSAddress

        if isinstance(address, LocalFSAddress):
            self._rdd.map(lambda x: hdfs_utils.serialize(x[0], x[1])).repartition(
                partitions
            ).saveAsTextFile(address.path)
            schema.update(self.schema)
            return

        raise NotImplementedError(
            f"address type {type(address)} not supported with spark backend"
        )

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
    def mapPartitions(
        self, func, use_previous_behavior=True, preserves_partitioning=False, **kwargs
    ):
        if use_previous_behavior is True:
            LOGGER.warning(
                f"please use `applyPartitions` instead of `mapPartitions` "
                f"if the previous behavior was expected. "
                f"The previous behavior will not work in future"
            )
            return self.applyPartitions(func)
        return from_rdd(
            self._rdd.mapPartitions(func, preservesPartitioning=preserves_partitioning)
        )

    @computing_profile
    def mapReducePartitions(self, mapper, reducer, **kwargs):
        return from_rdd(self._rdd.mapPartitions(mapper).reduceByKey(reducer))

    @computing_profile
    def applyPartitions(self, func, **kwargs):
        return from_rdd(_map_partitions(self._rdd, func))

    @computing_profile
    def glom(self, **kwargs):
        return from_rdd(_glom(self._rdd))

    @computing_profile
    def sample(
        self,
        *,
        fraction: typing.Optional[float] = None,
        num: typing.Optional[int] = None,
        seed=None,
    ):
        if fraction is not None:
            return from_rdd(
                self._rdd.sample(fraction=fraction, withReplacement=False, seed=seed)
            )

        if num is not None:
            return from_rdd(_exactly_sample(self._rdd, num, seed=seed))

        raise ValueError(
            f"exactly one of `fraction` or `num` required, fraction={fraction}, num={num}"
        )

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
        #         return iter(self._rdd.collect())
        return self._rdd.toLocalIterator()

    @computing_profile
    def take(self, n=1, **kwargs):
        _value = self._rdd.take(n)
        if kwargs.get("filter", False):
            self._rdd = self._rdd.filter(lambda xy: xy not in [_xy for _xy in _value])
        return _value

    @computing_profile
    def first(self, **kwargs):
        return self.take(1)[0]

    @computing_profile
    def count(self, **kwargs):
        if self._count is None:
            self._count = self._rdd.count()
        return self._count

    @computing_profile
    def join(self, other: "Table", func=None, **kwargs):
        return from_rdd(_join(self._rdd, other._rdd, func=func))

    @computing_profile
    def subtractByKey(self, other: "Table", **kwargs):
        return from_rdd(_subtract_by_key(self._rdd, other._rdd))

    @computing_profile
    def union(self, other: "Table", func=None, **kwargs):
        return from_rdd(_union(self._rdd, other._rdd, func))


def from_hdfs(paths: str, partitions, in_serialized=True, id_delimiter=None):
    # noinspection PyPackageRequirements
    from pyspark import SparkContext

    sc = SparkContext.getOrCreate()
    fun = hdfs_utils.deserialize if in_serialized else lambda x: (x.partition(id_delimiter)[0],
                                                                  x.partition(id_delimiter)[2])
    rdd = materialize(
        sc.textFile(paths, partitions)
        .map(fun)
        .repartition(partitions)
    )
    return Table(rdd=rdd)


def from_localfs(paths: str, partitions, in_serialized=True, id_delimiter=None):
    # noinspection PyPackageRequirements
    from pyspark import SparkContext

    sc = SparkContext.getOrCreate()
    fun = hdfs_utils.deserialize if in_serialized else lambda x: (x.partition(id_delimiter)[0],
                                                                  x.partition(id_delimiter)[2])
    rdd = materialize(
        sc.textFile(paths, partitions)
        .map(fun)
        .repartition(partitions)
    )
    return Table(rdd=rdd)


def from_hive(tb_name, db_name, partitions):
    from pyspark.sql import SparkSession

    session = SparkSession.builder.enableHiveSupport().getOrCreate()
    rdd = materialize(
        session.sql(f"select * from {db_name}.{tb_name}")
        .rdd.map(hive_utils.from_row)
        .repartition(partitions)
    )
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


def _exactly_sample(rdd, num: int, seed: int):
    split_size = rdd.mapPartitionsWithIndex(
        lambda s, it: [(s, sum(1 for _ in it))]
    ).collectAsMap()
    total = sum(split_size.values())

    if num > total:
        raise ValueError(f"not enough data to sample, own {total} but required {num}")
    # random the size of each split
    sampled_size = {}
    for split, size in split_size.items():
        sampled_size[split] = hypergeom.rvs(M=total, n=size, N=num)
        total = total - size
        num = num - sampled_size[split]

    return rdd.mapPartitionsWithIndex(
        _ReservoirSample(split_sample_size=sampled_size, seed=seed).func,
        preservesPartitioning=True,
    )


class _ReservoirSample(RDDSamplerBase):
    def __init__(self, split_sample_size, seed):
        RDDSamplerBase.__init__(self, False, seed)
        self._split_sample_size = split_sample_size
        self._counter = 0
        self._sample = []

    def func(self, split, iterator):
        self.initRandomGenerator(split)
        size = self._split_sample_size[split]
        for obj in iterator:
            self._counter += 1
            if len(self._sample) < size:
                self._sample.append(obj)
                continue

            randint = self._random.randint(1, self._counter)
            if randint <= size:
                self._sample[randint - 1] = obj

        return self._sample


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
    if func is None:
        return rdd.union(other).coalesce(num_partition)
    else:

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

    return rdd.mapPartitionsWithIndex(_func, preservesPartitioning=False)
