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

import abc
import logging
import random
from typing import Any, Callable, Tuple, Iterable, Generic, TypeVar, Optional

from fate.arch.computing.partitioners import get_partitioner_by_type
from fate.arch.computing.serdes import get_serdes_by_type
from fate.arch.trace import auto_trace
from fate.arch.trace import computing_profile as _compute_info
from fate.arch.unify import URI

logger = logging.getLogger(__name__)


K = TypeVar("K")
V = TypeVar("V")

_level = 0


def _add_padding(message, count):
    padding = " " * count
    lines = message.split("\n")
    padded_lines = [padding + line for line in lines]
    return "\n".join(padded_lines)


# def _compute_info(func):
#     return func
#
#     # @functools.wraps(func)
#     # def wrapper(*args, **kwargs):
#     #     global _level
#     #     logger.debug(_add_padding(f"computing enter {func.__name__}", _level * 2))
#     #     try:
#     #         _level += 1
#     #         stacks = _add_padding("".join(traceback.format_stack(limit=5)[:-1]), _level * 2)
#     #         logger.debug(f'{_add_padding("stack:", _level * 2)}\n{stacks}')
#     #         return func(*args, **kwargs)
#     #     finally:
#     #         _level -= 1
#     #         logger.debug(f"{' ' * _level}computing exit {func.__name__}")
#     #
#     # return wrapper


class KVTableContext:
    def _info(self):
        return {}

    def _load(self, uri: URI, schema: dict, options: dict):
        raise NotImplementedError(f"{self.__class__.__name__}._load")

    def _parallelize(
        self,
        data,
        total_partitions,
        key_serdes,
        key_serdes_type,
        value_serdes,
        value_serdes_type,
        partitioner,
        partitioner_type,
    ):
        raise NotImplementedError(f"{self.__class__.__name__}._parallelize")

    def _destroy(self):
        raise NotImplementedError(f"{self.__class__.__name__}.destroy")

    def info(self):
        return self._info()

    def load(self, uri: URI, schema: dict, options: dict = None):
        return self._load(
            uri=uri,
            schema=schema,
            options=options,
        )

    @_compute_info
    def parallelize(
        self, data, include_key=True, partition=None, key_serdes_type=0, value_serdes_type=0, partitioner_type=0
    ) -> "KVTable":
        key_serdes = get_serdes_by_type(key_serdes_type)
        value_serdes = get_serdes_by_type(value_serdes_type)
        partitioner = get_partitioner_by_type(partitioner_type)
        if partition is None:
            partition = 1
        if not include_key:
            data = ((key_serdes.serialize(i), value_serdes.serialize(v)) for i, v in enumerate(data))
        else:
            data = ((key_serdes.serialize(k), value_serdes.serialize(v)) for k, v in data)
        return self._parallelize(
            data=data,
            total_partitions=partition,
            key_serdes=key_serdes,
            key_serdes_type=key_serdes_type,
            value_serdes=value_serdes,
            value_serdes_type=value_serdes_type,
            partitioner=partitioner,
            partitioner_type=partitioner_type,
        )

    def destroy(self):
        self._destroy()


class KVTable(Generic[K, V]):
    def __init__(self, key_serdes_type, value_serdes_type, partitioner_type, num_partitions):
        self.key_serdes_type = key_serdes_type
        self.value_serdes_type = value_serdes_type
        self.partitioner_type = partitioner_type
        self.num_partitions = num_partitions

        self._count_cache = None
        self._key_serdes = None
        self._value_serdes = None
        self._partitioner = None

        self._schema = {}

        self._is_federated_received = False

    def mask_federated_received(self):
        self._is_federated_received = True

    def __getstate__(self):
        pass

    def __reduce__(self):
        raise NotImplementedError("Table is not pickleable, please don't do this or it may cause unexpected error")

    def __del__(self):
        if self._is_federated_received:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"destroying federated received table {self}")
            self.destroy()

    @property
    def schema(self):
        return self._schema

    @schema.setter
    def schema(self, schema):
        self._schema = schema

    @abc.abstractmethod
    def _save(self, uri: URI, schema, options: dict):
        raise NotImplementedError(f"{self.__class__.__name__}._save")

    @abc.abstractmethod
    def _drop_num(self, num: int, partitioner):
        raise NotImplementedError(f"{self.__class__.__name__}._drop_num")

    @abc.abstractmethod
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
        raise NotImplementedError(f"{self.__class__.__name__}._map_reduce_partitions_with_index")

    @abc.abstractmethod
    def _binary_sorted_map_partitions_with_index(
        self,
        other: "KVTable",
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
        raise NotImplementedError(f"{self.__class__.__name__}._subtract_by_key")

    @abc.abstractmethod
    def _reduce(self, func: Callable[[V, V], V]):
        raise NotImplementedError(f"{self.__class__.__name__}._reduce")

    @abc.abstractmethod
    def _collect(self):
        raise NotImplementedError(f"{self.__class__.__name__}._collect")

    @abc.abstractmethod
    def _take(self, n=1, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}._take")

    @abc.abstractmethod
    def _count(self):
        raise NotImplementedError(f"{self.__class__.__name__}._count")

    @abc.abstractmethod
    def _destroy(self):
        raise NotImplementedError(f"{self.__class__.__name__}.destroy")

    @property
    def key_serdes(self):
        if self._key_serdes is None:
            self._key_serdes = get_serdes_by_type(self.key_serdes_type)
        return self._key_serdes

    @property
    def value_serdes(self):
        if self._value_serdes is None:
            self._value_serdes = get_serdes_by_type(self.value_serdes_type)
        return self._value_serdes

    @property
    def partitioner(self):
        if self._partitioner is None:
            self._partitioner = get_partitioner_by_type(self.partitioner_type)
        return self._partitioner

    def __str__(self):
        return f"<{self.__class__.__name__} \
        key_serdes_type={self.key_serdes_type}, \
         value_serdes_type={self.value_serdes_type}, \
         partitioner_type={self.partitioner_type}>"

    def destroy(self):
        self._destroy()

    @auto_trace
    @_compute_info
    def map_reduce_partitions_with_index(
        self,
        map_partition_op: Callable[[int, Iterable], Iterable],
        reduce_partition_op: Callable[[Any, Any], Any] = None,
        shuffle=True,
        output_key_serdes_type=None,
        output_value_serdes_type=None,
        output_partitioner_type=None,
        output_num_partitions=None,
    ):
        return self._map_reduce_partitions_with_index(
            map_partition_op=map_partition_op,
            reduce_partition_op=reduce_partition_op,
            shuffle=shuffle,
            output_key_serdes_type=output_key_serdes_type,
            output_value_serdes_type=output_value_serdes_type,
            output_partitioner_type=output_partitioner_type,
            output_num_partitions=output_num_partitions,
        )

    def _map_reduce_partitions_with_index(
        self,
        map_partition_op: Callable[[int, Iterable], Iterable],
        reduce_partition_op: Callable[[Any, Any], Any] = None,
        shuffle=True,
        output_key_serdes_type=None,
        output_value_serdes_type=None,
        output_partitioner_type=None,
        output_num_partitions=None,
    ):
        if not shuffle and reduce_partition_op is not None:
            raise ValueError("when shuffle is False, it is not allowed to specify reduce_partition_op")
        if output_key_serdes_type is None:
            output_key_serdes_type = self.key_serdes_type
        if output_value_serdes_type is None:
            output_value_serdes_type = self.value_serdes_type
        if output_partitioner_type is None:
            output_partitioner_type = self.partitioner_type
        if output_num_partitions is None:
            output_num_partitions = self.num_partitions
        input_key_serdes = self.key_serdes
        input_value_serdes = self.value_serdes
        input_partitioner = self.partitioner
        output_key_serdes = get_serdes_by_type(output_key_serdes_type)
        output_value_serdes = get_serdes_by_type(output_value_serdes_type)
        output_partitioner = get_partitioner_by_type(output_partitioner_type)
        return self._impl_map_reduce_partitions_with_index(
            map_partition_op=_lifted_mpwi_map_to_serdes(
                map_partition_op, self.key_serdes, self.value_serdes, output_key_serdes, output_value_serdes
            ),
            reduce_partition_op=_lifted_mpwi_reduce_to_serdes(reduce_partition_op, output_value_serdes),
            shuffle=shuffle,
            input_key_serdes=input_key_serdes,
            input_key_serdes_type=self.key_serdes_type,
            input_value_serdes=input_value_serdes,
            input_value_serdes_type=self.value_serdes_type,
            input_partitioner=input_partitioner,
            input_partitioner_type=self.partitioner_type,
            output_key_serdes=output_key_serdes,
            output_key_serdes_type=output_key_serdes_type,
            output_value_serdes=output_value_serdes,
            output_value_serdes_type=output_value_serdes_type,
            output_partitioner=output_partitioner,
            output_partitioner_type=output_partitioner_type,
            output_num_partitions=output_num_partitions,
        )

    @auto_trace
    @_compute_info
    def mapPartitionsWithIndexNoSerdes(
        self,
        map_partition_op: Callable[[int, Iterable[Tuple[bytes, bytes]]], Iterable[Tuple[bytes, bytes]]],
        shuffle=False,
        output_key_serdes_type=None,
        output_value_serdes_type=None,
        output_partitioner_type=None,
    ):
        """
        caller should guarantee that the output of map_partition_op is a generator of (bytes, bytes)
        if shuffle is False, caller should guarantee that the output of map_partition_op is in the same partition
        according to the given output_partitioner_type and output_key_serdes_type and output_value_serdes_type

        this method is used to avoid unnecessary serdes/deserdes in message queue federation
        """
        return self._impl_map_reduce_partitions_with_index(
            map_partition_op=map_partition_op,
            reduce_partition_op=None,
            shuffle=shuffle,
            input_key_serdes=get_serdes_by_type(self.key_serdes_type),
            input_key_serdes_type=self.key_serdes_type,
            input_value_serdes=get_serdes_by_type(self.value_serdes_type),
            input_value_serdes_type=self.value_serdes_type,
            input_partitioner=get_partitioner_by_type(self.partitioner_type),
            input_partitioner_type=self.partitioner_type,
            output_key_serdes=get_serdes_by_type(output_key_serdes_type),
            output_key_serdes_type=output_key_serdes_type,
            output_value_serdes=get_serdes_by_type(output_value_serdes_type),
            output_value_serdes_type=output_value_serdes_type,
            output_partitioner=get_partitioner_by_type(output_partitioner_type),
            output_partitioner_type=output_partitioner_type,
            output_num_partitions=self.num_partitions,
        )

    @auto_trace
    @_compute_info
    def mapPartitionsWithIndex(
        self,
        map_partition_op: Callable[[int, Iterable], Iterable],
        output_key_serdes_type=None,
        output_value_serdes_type=None,
        output_partitioner_type=None,
    ):
        return self._map_reduce_partitions_with_index(
            map_partition_op=map_partition_op,
            shuffle=True,
            output_key_serdes_type=output_key_serdes_type,
            output_value_serdes_type=output_value_serdes_type,
            output_partitioner_type=output_partitioner_type,
        )

    @auto_trace
    @_compute_info
    def mapReducePartitions(
        self,
        map_partition_op: Callable[[Iterable], Iterable],
        reduce_partition_op: Callable[[Any, Any], Any] = None,
        shuffle=True,
        output_key_serdes_type=None,
        output_value_serdes_type=None,
        output_partitioner_type=None,
    ):
        return self._map_reduce_partitions_with_index(
            map_partition_op=_lifted_map_reduce_partitions_to_mpwi(map_partition_op),
            reduce_partition_op=reduce_partition_op,
            shuffle=shuffle,
            output_key_serdes_type=output_key_serdes_type,
            output_value_serdes_type=output_value_serdes_type,
            output_partitioner_type=output_partitioner_type,
        )

    @auto_trace
    @_compute_info
    def applyPartitions(self, func, output_value_serdes_type=None):
        return self._map_reduce_partitions_with_index(
            map_partition_op=_lifted_apply_partitions_to_mpwi(func),
            shuffle=False,
            output_key_serdes_type=self.key_serdes_type,
            output_value_serdes_type=output_value_serdes_type,
        )

    @auto_trace
    @_compute_info
    def mapPartitions(
        self, func, use_previous_behavior=False, preserves_partitioning=False, output_value_serdes_type=None
    ):
        if use_previous_behavior:
            raise NotImplementedError("use_previous_behavior is not supported")
        return self._map_reduce_partitions_with_index(
            map_partition_op=_lifted_map_partitions_to_mpwi(func),
            shuffle=not preserves_partitioning,
            output_key_serdes_type=self.key_serdes_type,
            output_value_serdes_type=output_value_serdes_type,
        )

    @auto_trace
    @_compute_info
    def map(
        self,
        map_op: Callable[[Any, Any], Tuple[Any, Any]],
        output_key_serdes_type=None,
        output_value_serdes_type=None,
        output_partitioner_type=None,
    ):
        return self._map_reduce_partitions_with_index(
            _lifted_map_to_mpwi(map_op),
            shuffle=True,
            output_key_serdes_type=output_key_serdes_type,
            output_value_serdes_type=output_value_serdes_type,
            output_partitioner_type=output_partitioner_type,
        )

    @auto_trace
    @_compute_info
    def mapValues(self, map_value_op: Callable[[Any], Any], output_value_serdes_type=None):
        return self._map_reduce_partitions_with_index(
            _lifted_map_values_to_mpwi(map_value_op),
            shuffle=False,
            output_key_serdes_type=self.key_serdes_type,
            output_value_serdes_type=output_value_serdes_type,
        )

    @auto_trace
    @_compute_info
    def copy(self):
        return self.mapValues(lambda x: x, output_value_serdes_type=self.value_serdes_type)

    @auto_trace
    @_compute_info
    def flatMap(
        self,
        flat_map_op: Callable[[Any, Any], Iterable[Tuple[Any, Any]]],
        output_key_serdes_type=None,
        output_value_serdes_type=None,
    ):
        return self._map_reduce_partitions_with_index(
            _lifted_flat_map_to_mpwi(flat_map_op),
            shuffle=True,
            output_key_serdes_type=output_key_serdes_type,
            output_value_serdes_type=output_value_serdes_type,
        )

    @auto_trace
    @_compute_info
    def filter(self, filter_op: Callable[[Any], bool]):
        return self._map_reduce_partitions_with_index(
            lambda i, x: ((k, v) for k, v in x if filter_op(v)),
            shuffle=False,
            output_key_serdes_type=self.key_serdes_type,
            output_value_serdes_type=self.value_serdes_type,
        )

    def _sample(self, fraction, seed=None) -> "KVTable":
        return self._map_reduce_partitions_with_index(
            _lifted_sample_to_mpwi(fraction, seed),
            shuffle=False,
            output_key_serdes_type=self.key_serdes_type,
            output_value_serdes_type=self.value_serdes_type,
        )

    @auto_trace
    @_compute_info
    def collect(self):
        for k, v in self._collect():
            yield self.key_serdes.deserialize(k), self.value_serdes.deserialize(v)

    @auto_trace
    @_compute_info
    def take(self, n):
        return [(self.key_serdes.deserialize(k), self.value_serdes.deserialize(v)) for k, v in self._take(n)]

    @auto_trace
    @_compute_info
    def first(self):
        resp = self.take(n=1)
        if len(resp) < 1:
            raise RuntimeError("table is empty")
        return resp[0]

    @auto_trace
    @_compute_info
    def reduce(self, func: Callable[[V, V], V]) -> V:
        return self.value_serdes.deserialize(self._reduce(_lifted_reduce_to_serdes(func, self.value_serdes)))

    @auto_trace
    @_compute_info
    def count(self) -> int:
        if self._count_cache is None:
            self._count_cache = self._count()
        return self._count_cache

    @auto_trace
    @_compute_info
    def join(
        self,
        other: "KVTable",
        merge_op: Callable[[V, V], V] = lambda x, y: x,
        output_value_serdes_type=None,
    ):
        return self.binarySortedMapPartitionsWithIndex(
            other,
            _lifted_join_merge_to_sbmpwi(merge_op),
            output_value_serdes_type=output_value_serdes_type,
        )

    @auto_trace
    @_compute_info
    def union(self, other, merge_op: Callable[[V, V], V] = lambda x, y: x, output_value_serdes_type=None):
        return self.binarySortedMapPartitionsWithIndex(
            other,
            _lifted_union_merge_to_sbmpwi(merge_op),
            output_value_serdes_type=output_value_serdes_type,
        )

    @auto_trace
    @_compute_info
    def subtractByKey(self, other: "KVTable"):
        return self.binarySortedMapPartitionsWithIndex(
            other,
            _lifted_subtract_by_key_to_sbmpwi(),
            output_value_serdes_type=self.value_serdes_type,
        )

    def binarySortedMapPartitionsWithIndex(
        self,
        other: "KVTable",
        binary_sorted_map_partitions_with_index_op: Callable[[int, Iterable, Iterable], Iterable],
        output_value_serdes_type=None,
    ):
        if output_value_serdes_type is None:
            output_value_serdes_type = self.value_serdes_type
        assert self.key_serdes_type == other.key_serdes_type, "key_serdes_type must be the same"
        assert self.partitioner_type == other.partitioner_type, "partitioner_type must be the same"
        output_value_serdes = get_serdes_by_type(output_value_serdes_type)

        # makes partition alignment:
        #   self.num_partitions == other.num_partitions
        #   self.partitioner_type == other.partitioner_type
        first, second = self.repartition_with(other)

        # apply binary_sorted_map_partitions_with_index_op
        return first._binary_sorted_map_partitions_with_index(
            other=second,
            binary_map_partitions_with_index_op=_lifted_sorted_binary_map_partitions_with_index_to_serdes(
                binary_sorted_map_partitions_with_index_op,
                first.value_serdes,
                second.value_serdes,
                output_value_serdes,
            ),
            key_serdes=first.key_serdes,
            key_serdes_type=first.key_serdes_type,
            partitioner=first.partitioner,
            partitioner_type=first.partitioner_type,
            first_input_value_serdes=first.value_serdes,
            first_input_value_serdes_type=first.value_serdes_type,
            second_input_value_serdes=second.value_serdes,
            second_input_value_serdes_type=second.value_serdes_type,
            output_value_serdes=output_value_serdes,
            output_value_serdes_type=output_value_serdes_type,
        )

    @auto_trace
    @_compute_info
    def repartition(self, num_partitions, partitioner_type=None, key_serdes_type=None) -> "KVTable":
        if (
            self.num_partitions == num_partitions
            and ((partitioner_type is None) or self.partitioner_type == partitioner_type)
            and ((key_serdes_type is None) or key_serdes_type == self.key_serdes_type)
        ):
            return self
        if partitioner_type is None:
            partitioner_type = self.partitioner_type
        if self.partitioner_type != partitioner_type:
            output_partitioner = get_partitioner_by_type(partitioner_type)
        else:
            output_partitioner = self.partitioner

        if key_serdes_type is None:
            key_serdes_type = self.key_serdes_type
        if key_serdes_type != self.key_serdes_type:
            output_key_serdes = get_serdes_by_type(key_serdes_type)
        else:
            output_key_serdes = self.key_serdes

        if self.key_serdes_type == key_serdes_type and self.partitioner_type == partitioner_type:
            mapper = lambda i, x: x
        else:
            mapper = _lifted_map_to_io_serdes(
                lambda i, x: x,
                self.key_serdes,
                self.value_serdes,
                output_key_serdes,
                self.value_serdes,
            )
        return self._impl_map_reduce_partitions_with_index(
            map_partition_op=mapper,
            reduce_partition_op=None,
            shuffle=True,
            input_key_serdes=self.key_serdes,
            input_key_serdes_type=self.key_serdes_type,
            input_value_serdes=self.value_serdes,
            input_value_serdes_type=self.value_serdes_type,
            input_partitioner=self.partitioner,
            input_partitioner_type=self.partitioner_type,
            output_key_serdes=output_key_serdes,
            output_key_serdes_type=key_serdes_type,
            output_value_serdes=self.value_serdes,
            output_value_serdes_type=self.value_serdes_type,
            output_partitioner=output_partitioner,
            output_partitioner_type=partitioner_type,
            output_num_partitions=num_partitions,
        )

    @auto_trace
    @_compute_info
    def repartition_with(self, other: "KVTable") -> Tuple["KVTable", "KVTable"]:
        if self.partitioner_type == other.partitioner_type and self.num_partitions == other.num_partitions:
            return self, other
        if self.num_partitions > other.num_partitions:
            return self, other.repartition(self.num_partitions, self.partitioner_type)
        else:
            return self.repartition(other.num_partitions, other.partitioner_type), other

    @auto_trace
    @_compute_info
    def save(self, uri: URI, schema, options: dict = None):
        options = options or {}
        if (partition := options.get("partition")) is not None and partition != self.num_partitions:
            self.repartition(partition)._save(uri, schema, options)
        else:
            self._save(uri, schema, options)
        schema.update(self.schema)

    @auto_trace
    @_compute_info
    def sample(
        self,
        *,
        fraction: Optional[float] = None,
        num: Optional[int] = None,
        seed=None,
    ):
        if fraction is None and num is None:
            raise ValueError("either fraction or num must be specified")

        if fraction is not None:
            return self._sample(fraction=fraction, seed=seed)

        if num is not None:
            total = self.count()
            if num > total:
                raise ValueError(f"not enough data to sample, own {total} but required {num}")

            frac = num / float(total)
            while True:
                sampled_table = self._sample(fraction=frac, seed=seed)
                sampled_count = sampled_table.count()
                if sampled_count < num:
                    frac *= 1.1
                else:
                    break
            if sampled_count == num:
                return sampled_table
            else:
                return sampled_table._drop_num(sampled_count - num, self.partitioner)


def _lifted_map_to_io_serdes(_f, input_key_serdes, input_value_serdes, output_key_serdes, output_value_serdes):
    def _lifted(_index, _iter):
        for out_k, out_v in _f(_index, _serdes_wrapped_generator(_iter, input_key_serdes, input_value_serdes)):
            yield output_key_serdes.serialize(out_k), output_value_serdes.serialize(out_v)

    return _lifted


def _serdes_wrapped_generator(_iter, key_serdes, value_serdes):
    for k, v in _iter:
        yield key_serdes.deserialize(k), value_serdes.deserialize(v)


def _value_serdes_wrapped_generator(_iter, value_serdes):
    for k, v in _iter:
        yield k, value_serdes.deserialize(v)


def _lifted_mpwi_map_to_serdes(_f, input_key_serdes, input_value_serdes, output_key_serdes, output_value_serdes):
    def _lifted(_index, _iter):
        for out_k, out_v in _f(_index, _serdes_wrapped_generator(_iter, input_key_serdes, input_value_serdes)):
            yield output_key_serdes.serialize(out_k), output_value_serdes.serialize(out_v)

    return _lifted


def _lifted_mpwi_reduce_to_serdes(_f, output_value_serdes):
    if _f is None:
        return None

    def _lifted(x, y):
        return output_value_serdes.serialize(
            _f(
                output_value_serdes.deserialize(x),
                output_value_serdes.deserialize(y),
            )
        )

    return _lifted


def _lifted_map_values_to_mpwi(map_value_op: Callable[[Any], Any]):
    def _lifted(_index, _iter):
        for _k, _v in _iter:
            yield _k, map_value_op(_v)

    return _lifted


def _lifted_map_to_mpwi(map_op: Callable[[Any, Any], Tuple[Any, Any]]):
    def _lifted(_index, _iter):
        for _k, _v in _iter:
            yield map_op(_k, _v)

    return _lifted


def _lifted_map_reduce_partitions_to_mpwi(map_partition_op: Callable[[Iterable], Iterable]):
    def _lifted(_index, _iter):
        return map_partition_op(_iter)

    return _lifted


def _get_generator_with_last_key(_iter):
    cache = [None]

    def _generator():
        for k, v in _iter:
            cache[0] = k
            yield k, v

    return _generator, cache


def _lifted_apply_partitions_to_mpwi(apply_partition_op: Callable[[Iterable], Any]):
    def _lifted(_index, _iter):
        _iter_set_cache, _cache = _get_generator_with_last_key(_iter)
        value = apply_partition_op(_iter_set_cache())
        key = _cache[0]
        if key is None:
            return []
        return [(key, value)]

    return _lifted


def _lifted_map_partitions_to_mpwi(map_partition_op: Callable[[Iterable], Iterable]):
    def _lifted(_index, _iter):
        return map_partition_op(_iter)

    return _lifted


def _lifted_flat_map_to_mpwi(flat_map_op: Callable[[Any, Any], Iterable[Tuple[Any, Any]]]):
    def _lifted(_index, _iter):
        for _k, _v in _iter:
            yield from flat_map_op(_k, _v)

    return _lifted


def _lifted_sample_to_mpwi(fraction, seed=None):
    def _lifted(_index, _iter):
        # TODO: should we use the same seed for all partitions?
        random_state = random.Random(seed)
        for _k, _v in _iter:
            if random_state.random() < fraction:
                yield _k, _v

    return _lifted


def _lifted_reduce_to_serdes(reduce_op, value_serdes):
    def _lifted(x, y):
        return value_serdes.serialize(
            reduce_op(
                value_serdes.deserialize(x),
                value_serdes.deserialize(y),
            )
        )

    return _lifted


def _lifted_sorted_binary_map_partitions_with_index_to_serdes(
    _f, left_value_serdes, right_value_serdes, output_value_serdes
):
    def _lifted(_index, left_iter, right_iter):
        for out_k_bytes, out_v in _f(
            _index,
            _value_serdes_wrapped_generator(left_iter, left_value_serdes),
            _value_serdes_wrapped_generator(right_iter, right_value_serdes),
        ):
            yield out_k_bytes, output_value_serdes.serialize(out_v)

    return _lifted


def _lifted_join_merge_to_sbmpwi(join_merge_op):
    def _lifted(_index, _left_iter, _right_iter):
        return _merge_intersecting_keys(_left_iter, _right_iter, join_merge_op)

    return _lifted


def _lifted_union_merge_to_sbmpwi(join_merge_op):
    def _lifted(_index, _left_iter, _right_iter):
        return _merge_union_keys(_left_iter, _right_iter, join_merge_op)

    return _lifted


def _lifted_subtract_by_key_to_sbmpwi():
    def _lifted(_index, _left_iter, _right_iter):
        return _subtract_by_key(_left_iter, _right_iter)

    return _lifted


def _merge_intersecting_keys(iter1, iter2, merge_op):
    try:
        item1 = next(iter1)
        item2 = next(iter2)
    except StopIteration:
        return

    while True:
        key1, value1 = item1
        key2, value2 = item2

        if key1 == key2:
            yield key1, merge_op(value1, value2)
            try:
                item1 = next(iter1)
                item2 = next(iter2)
            except StopIteration:
                break
        elif key1 < key2:
            try:
                item1 = next(iter1)
            except StopIteration:
                break
        else:  # key1 > key2
            try:
                item2 = next(iter2)
            except StopIteration:
                break


def _merge_union_keys(iter1, iter2, merge_op):
    try:
        item1 = next(iter1)
    except StopIteration:
        item1 = None

    try:
        item2 = next(iter2)
    except StopIteration:
        item2 = None

    if item1 is None and item2 is None:
        return

    while item1 is not None and item2 is not None:
        key1, value1 = item1
        key2, value2 = item2

        if key1 == key2:
            yield key1, merge_op(value1, value2)
            try:
                item1 = next(iter1)
            except StopIteration:
                item1 = None
            try:
                item2 = next(iter2)
            except StopIteration:
                item2 = None
        elif key1 < key2:
            yield key1, value1
            try:
                item1 = next(iter1)
            except StopIteration:
                item1 = None
        else:  # key1 > key2
            yield key2, value2
            try:
                item2 = next(iter2)
            except StopIteration:
                item2 = None

    if item1 is not None:
        yield item1
        yield from iter1
    elif item2 is not None:
        yield item2
        yield from iter2


def _subtract_by_key(iter1, iter2):
    try:
        item1 = next(iter1)
    except StopIteration:
        return

    try:
        item2 = next(iter2)
    except StopIteration:
        yield item1
        yield from iter1
        return

    while item1 is not None and item2 is not None:
        key1, value1 = item1
        key2, value2 = item2

        if key1 == key2:
            try:
                item1 = next(iter1)
            except StopIteration:
                item1 = None
            try:
                item2 = next(iter2)
            except StopIteration:
                item2 = None
        elif key1 < key2:
            yield item1
            try:
                item1 = next(iter1)
            except StopIteration:
                item1 = None
        else:  # key1 > key2
            try:
                item2 = next(iter2)
            except StopIteration:
                item2 = None

    if item1 is not None:
        yield item1
        yield from iter1


def is_table(v):
    return isinstance(v, KVTable)
