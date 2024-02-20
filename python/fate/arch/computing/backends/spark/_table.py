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

import logging
import struct
import typing
from typing import Callable, Iterable, Tuple, Optional, Any

import pyspark
from pyspark.rddsampler import RDDSamplerBase

from fate.arch import URI
from fate.arch.computing.api import KVTable, ComputingEngine, K, V
from fate.arch.computing.api._table import _lifted_reduce_to_serdes, get_serdes_by_type
from fate.arch.computing.partitioners import get_partitioner_by_type
from fate.arch.trace import auto_trace
from fate.arch.trace import computing_profile as _compute_info
from ._materialize import materialize, unmaterialize

LOGGER = logging.getLogger(__name__)


class HDFSCoder:
    @staticmethod
    def encode(key: bytes, value: bytes):
        size = struct.pack(">Q", len(key))
        return (size + key + value).hex()

    @staticmethod
    def decode(data: str):
        data = bytes.fromhex(data)
        size = struct.unpack(">Q", data[:8])[0]
        key = data[8 : 8 + size]
        value = data[8 + size :]
        return key, value


class HiveCoder:
    @staticmethod
    def encode(key: bytes, value: bytes):
        from pyspark.sql import Row

        return Row(key=key.hex(), value=value.hex())

    @staticmethod
    def ecode(r):
        return bytes.fromhex(r.key), bytes.fromhex(r.value)


class Table(KVTable):
    def __init__(self, rdd: pyspark.RDD, key_serdes_type, value_serdes_type, partitioner_type):
        self._rdd = rdd
        self._engine = ComputingEngine.SPARK

        super().__init__(
            key_serdes_type=key_serdes_type,
            value_serdes_type=value_serdes_type,
            partitioner_type=partitioner_type,
            num_partitions=rdd.getNumPartitions(),
        )

    @property
    def rdd(self):
        return self._rdd

    def _binary_sorted_map_partitions_with_index(
        self,
        other: "Table",
        binary_map_partitions_with_index_op: Callable[[int, Iterable, Iterable], Iterable],
        key_serdes,
        key_serdes_type,
        partitioner,
        partitioner_type,
        first_input_value_serdes,
        first_input_value_serdes_type,
        second_input_value_serdes,
        second_input_value_serdes_type,
        output_value_serdes,
        output_value_serdes_type,
    ):
        raise NotImplementedError("binary sorted map partitions with index not supported in spark backend")

    @auto_trace
    @_compute_info
    def join(
        self,
        other: "Table",
        merge_op: Callable[[V, V], V] = None,
        output_value_serdes_type=None,
    ):
        num_partitions = max(self.num_partitions, other.num_partitions)
        rdd = self.rdd.join(other.rdd, numPartitions=num_partitions)
        if merge_op is not None:
            op = _lifted_reduce_to_serdes(merge_op, get_serdes_by_type(self.value_serdes_type))
            rdd = rdd.mapValues(lambda x: op(x[0], x[1]))
        return from_rdd(
            rdd=rdd,
            key_serdes_type=self.key_serdes_type,
            value_serdes_type=output_value_serdes_type or self.value_serdes_type,
            partitioner_type=self.partitioner_type,
        )

    @auto_trace
    @_compute_info
    def union(self, other: "Table", merge_op: Callable[[V, V], V] = None, output_value_serdes_type=None):
        num_partitions = max(self.num_partitions, other.num_partitions)
        if merge_op is None:
            return from_rdd(
                self.rdd.union(other.rdd).coalesce(num_partitions),
                key_serdes_type=self.key_serdes_type,
                value_serdes_type=output_value_serdes_type or self.value_serdes_type,
                partitioner_type=self.partitioner_type,
            )

        op = _lifted_reduce_to_serdes(merge_op, get_serdes_by_type(self.value_serdes_type))
        return from_rdd(
            self.rdd.union(other.rdd).reduceByKey(op, numPartitions=num_partitions),
            key_serdes_type=self.key_serdes_type,
            value_serdes_type=output_value_serdes_type or self.value_serdes_type,
            partitioner_type=self.partitioner_type,
        )

    @auto_trace
    @_compute_info
    def subtractByKey(self, other: "Table", output_value_serdes_type=None):
        return from_rdd(
            self.rdd.subtractByKey(other.rdd, numPartitions=self.num_partitions),
            key_serdes_type=self.key_serdes_type,
            value_serdes_type=output_value_serdes_type or self.value_serdes_type,
            partitioner_type=self.partitioner_type,
        )

    def mapPartitionsWithIndexNoSerdes(
        self,
        map_partition_op: Callable[[int, Iterable[Tuple[bytes, bytes]]], Iterable[Tuple[bytes, bytes]]],
        shuffle=False,
        output_key_serdes_type=None,
        output_value_serdes_type=None,
        output_partitioner_type=None,
    ):
        # Note: since we use this method to send data to other parties, and if the engine in other side is not spark,
        # we should guarantee the data properly partitioned before we send each partition to other side.
        # So we should call _as_partitioned() before we call this method.
        # TODO: but if other side is also spark, we can skip _as_partitioned() to save time.
        return super().mapPartitionsWithIndexNoSerdes(
            map_partition_op=map_partition_op,
            shuffle=shuffle,
            output_key_serdes_type=output_key_serdes_type,
            output_value_serdes_type=output_value_serdes_type,
            output_partitioner_type=output_partitioner_type,
        )

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

    def _count(self):
        return self._rdd.count()

    def _take(self, n=1, **kwargs):
        _value = self._rdd.take(n)
        if kwargs.get("filter", False):
            self._rdd = self._rdd.filter(lambda xy: xy not in [_xy for _xy in _value])
        return _value

    def _collect(self, **kwargs):
        #         return iter(self.rdd.collect())
        return self._rdd.toLocalIterator()

    def _reduce(self, func, **kwargs):
        return self._rdd.values().reduce(func)

    def _drop_num(self, num: int, partitioner):
        raise NotImplementedError("drop num not supported in spark backend")

    def _impl_map_reduce_partitions_with_index(
        self,
        map_partition_op: Callable[[int, Iterable[Tuple[K, V]]], Iterable],
        reduce_partition_op: Optional[Callable[[Any, Any], Any]],
        shuffle: bool,
        input_key_serdes,
        input_key_serdes_type: int,
        input_value_serdes,
        input_value_serdes_type: int,
        input_partitioner,
        input_partitioner_type: int,
        output_key_serdes,
        output_key_serdes_type: int,
        output_value_serdes,
        output_value_serdes_type: int,
        output_partitioner,
        output_partitioner_type: int,
        output_num_partitions: int,
    ) -> "KVTable":
        rdd = self.rdd.mapPartitionsWithIndex(map_partition_op)

        if reduce_partition_op is not None:
            rdd = rdd.reduceByKey(reduce_partition_op)
        return from_rdd(
            rdd=rdd,
            key_serdes_type=output_key_serdes_type,
            value_serdes_type=output_value_serdes_type,
            partitioner_type=output_partitioner_type,
        )

    @auto_trace
    @_compute_info
    def sample(
        self,
        *,
        fraction: typing.Optional[float] = None,
        num: typing.Optional[int] = None,
        seed=None,
    ):
        if fraction is not None:
            return from_rdd(self.rdd.sample(fraction=fraction, withReplacement=False, seed=seed))

        if num is not None:
            return from_rdd(_exactly_sample(self.rdd, num, seed=seed))

        raise ValueError(f"exactly one of `fraction` or `num` required, fraction={fraction}, num={num}")

    def _destroy(self):
        pass

    def _save(self, uri: URI, schema, options: dict):
        if options is None:
            options = {}
        partitions = options.get("partitions")
        if uri.scheme == "hdfs":
            table = self.rdd.map(lambda x: HDFSCoder.encode(x[0], x[1]))
            if partitions:
                table = table.repartition(partitions)
            table.saveAsTextFile(uri.original_uri)
            schema.update(self.schema)
            return

        if uri.scheme == "hive":
            table = self.rdd.map(lambda x: HiveCoder.encode(x[0], x[1]))
            if partitions:
                table = table.repartition(partitions)
            table.toDF().write.saveAsTable(uri.original_uri)
            schema.update(self.schema)
            return

        if uri.scheme == "file":
            table = self.rdd.map(lambda x: HDFSCoder.encode(x[0], x[1]))
            if partitions:
                table = table.repartition(partitions)
            table.saveAsTextFile(uri.path)
            schema.update(self.schema)
            return

        raise NotImplementedError(f"uri type {uri} not supported with spark backend")


def from_hdfs(paths: str, partitions, in_serialized=True, id_delimiter=None):
    # noinspection PyPackageRequirements
    from pyspark import SparkContext

    sc = SparkContext.getOrCreate()
    fun = HDFSCoder.decode if in_serialized else lambda x: (x.partition(id_delimiter)[0], x.partition(id_delimiter)[2])
    rdd = sc.textFile(paths, partitions).map(fun)
    if partitions is not None:
        rdd = rdd.repartition(partitions)

    return from_rdd(rdd=rdd)


def from_localfs(paths: str, partitions, in_serialized=True, id_delimiter=None):
    # noinspection PyPackageRequirements
    from pyspark import SparkContext

    sc = SparkContext.getOrCreate()
    fun = HDFSCoder.decode if in_serialized else lambda x: (x.partition(id_delimiter)[0], x.partition(id_delimiter)[2])

    rdd = sc.textFile(paths, partitions).map(fun).repartition(partitions)

    return from_rdd(rdd=rdd)


def from_hive(tb_name, db_name, partitions):
    from pyspark.sql import SparkSession

    session = SparkSession.builder.enableHiveSupport().getOrCreate()

    rdd = session.sql(f"select * from {db_name}.{tb_name}").rdd.map(HiveCoder.decode).repartition(partitions)

    return from_rdd(rdd=rdd)


def from_rdd(rdd, key_serdes_type=0, value_serdes_type=0, partitioner_type=0):
    partitioner = get_partitioner_by_type(partitioner_type)
    num_partitions = rdd.getNumPartitions()
    rdd = rdd.partitionBy(num_partitions, lambda x: partitioner(x, num_partitions))

    rdd = materialize(rdd)

    return Table(
        rdd=rdd,
        key_serdes_type=key_serdes_type,
        value_serdes_type=value_serdes_type,
        partitioner_type=partitioner_type,
    )


def _exactly_sample(rdd, num: int, seed: int):
    from scipy.stats import hypergeom

    split_size = rdd.mapPartitionsWithIndex(lambda s, it: [(s, sum(1 for _ in it))]).collectAsMap()
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
