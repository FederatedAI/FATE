import random
from typing import Any, Callable, Tuple, Iterable, Generic, TypeVar

from fate.arch.unify.serdes import get_serdes_by_type
from fate.arch.unify.partitioner import get_partitioner_by_type
from ..unify import URI


K = TypeVar("K")
V = TypeVar("V")


class KVTableContext:
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

    def load(self, uri: URI, schema: dict, options: dict = None):
        return self._load(
            uri=uri,
            schema=schema,
            options=options,
        )

    def _load(self, uri: URI, schema: dict, options: dict):
        raise NotImplementedError(f"{self.__class__.__name__}._load")


class KVTable(Generic[K, V]):
    def __init__(self, key_serdes_type, value_serdes_type, partitioner_type):
        self.key_serdes_type = key_serdes_type
        self.value_serdes_type = value_serdes_type
        self.partitioner_type = partitioner_type

        self._count_cache = None
        self._key_serdes = None
        self._value_serdes = None
        self._partitioner = None

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
        return f"<{self.__class__.__name__} key_serdes={self.key_serdes}, value_serdes={self.value_serdes}, partitioner={self.partitioner}>"

    def _map_reduce_partitions_with_index(
        self,
        map_partition_op: Callable[[int, Iterable], Iterable],
        reduce_partition_op: Callable[[Any, Any], Any],
        shuffle,
        output_key_serdes,
        output_key_serdes_type,
        output_value_serdes,
        output_value_serdes_type,
        output_partitioner,
        output_partitioner_type,
    ):
        raise NotImplementedError(f"{self.__class__.__name__}._map_reduce_partitions_with_index")

    def _collect(self):
        raise NotImplementedError(f"{self.__class__.__name__}._collect")

    def _take(self, n=1, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}._take")

    def _count(self):
        raise NotImplementedError(f"{self.__class__.__name__}._count")

    def map_reduce_partitions_with_index(
        self,
        map_partition_op: Callable[[int, Iterable], Iterable],
        reduce_partition_op: Callable[[Any, Any], Any] = None,
        shuffle=True,
        output_key_serdes=None,
        output_value_serdes=None,
        output_partitioner=None,
    ):
        if output_key_serdes is None:
            output_key_serdes = self.key_serdes
        if output_value_serdes is None:
            output_value_serdes = self.value_serdes
        if output_partitioner is None:
            output_partitioner = self.partitioner
        return self._map_reduce_partitions_with_index(
            map_partition_op=_lifted_mpwi_map_to_serdes(
                map_partition_op, self.key_serdes, self.value_serdes, output_key_serdes, output_value_serdes
            ),
            reduce_partition_op=_lifted_mpwi_reduce_to_serdes(reduce_partition_op, output_value_serdes),
            shuffle=shuffle,
            output_key_serdes=output_key_serdes,
            output_key_serdes_type=0,
            output_value_serdes=output_value_serdes,
            output_value_serdes_type=0,
            output_partitioner=output_partitioner,
            output_partitioner_type=0,
        )

    def mapPartitionsWithIndex(
        self,
        map_partition_op: Callable[[int, Iterable], Iterable],
        output_key_serdes=None,
        output_value_serdes=None,
        output_partitioner=None,
    ):
        return self.map_reduce_partitions_with_index(
            map_partition_op=map_partition_op,
            shuffle=True,
            output_key_serdes=output_key_serdes,
            output_value_serdes=output_value_serdes,
            output_partitioner=output_partitioner,
        )

    def mapReducePartitions(
        self,
        map_partition_op: Callable[[Iterable], Iterable],
        reduce_partition_op: Callable[[Any, Any], Any] = None,
        shuffle=True,
        output_key_serdes=None,
        output_value_serdes=None,
        output_partitioner=None,
    ):
        return self.map_reduce_partitions_with_index(
            map_partition_op=_lifted_map_reduce_partitions_to_mpwi(map_partition_op),
            reduce_partition_op=reduce_partition_op,
            shuffle=shuffle,
            output_key_serdes=output_key_serdes,
            output_value_serdes=output_value_serdes,
            output_partitioner=output_partitioner,
        )

    def applyPartitions(self, func, output_value_serdes=None):
        return self.map_reduce_partitions_with_index(
            map_partition_op=_lifted_apply_partitions_to_mpwi(func),
            shuffle=False,
            output_key_serdes=self.key_serdes,
            output_value_serdes=output_value_serdes,
        )

    def mapPartitions(self, func, use_previous_behavior=False, preserves_partitioning=False, output_value_serdes=None):
        if use_previous_behavior:
            raise NotImplementedError("use_previous_behavior is not supported")
        return self.map_reduce_partitions_with_index(
            map_partition_op=_lifted_map_partitions_to_mpwi(func),
            shuffle=not preserves_partitioning,
            output_key_serdes=self.key_serdes,
            output_value_serdes=output_value_serdes,
        )

    def map(
        self,
        map_op: Callable[[Any, Any], Tuple[Any, Any]],
        output_key_serdes=None,
        output_value_serdes=None,
        output_partitioner=None,
    ):
        return self.map_reduce_partitions_with_index(
            _lifted_map_to_mpwi(map_op),
            shuffle=True,
            output_key_serdes=output_key_serdes,
            output_value_serdes=output_value_serdes,
        )

    def mapValues(self, map_value_op: Callable[[Any], Any], output_value_serdes=None):
        return self.map_reduce_partitions_with_index(
            _lifted_map_values_to_mpwi(map_value_op),
            shuffle=False,
            output_key_serdes=self.key_serdes,
            output_value_serdes=output_value_serdes,
        )

    def copy(self):
        return self.mapValues(lambda x: x, output_value_serdes=self.value_serdes)

    def flatMap(
        self,
        flat_map_op: Callable[[Any, Any], Iterable[Tuple[Any, Any]]],
        output_key_serdes=None,
        output_value_serdes=None,
    ):
        return self.map_reduce_partitions_with_index(
            _lifted_flat_map_to_mpwi(flat_map_op),
            shuffle=True,
            output_key_serdes=output_key_serdes,
            output_value_serdes=output_value_serdes,
        )

    def filter(self, filter_op: Callable[[Any], bool]):
        return self.map_reduce_partitions_with_index(
            lambda i, x: ((k, v) for k, v in x if filter_op(v)),
            shuffle=False,
            output_key_serdes=self.key_serdes,
            output_value_serdes=self.value_serdes,
        )

    def _sample(self, fraction, seed=None):
        return self.map_reduce_partitions_with_index(
            _lifted_sample_to_mpwi(fraction, seed),
            shuffle=False,
            output_key_serdes=self.key_serdes,
            output_value_serdes=self.value_serdes,
        )

    def collect(self):
        for k, v in self._collect():
            yield self.key_serdes.deserialize(k), self.value_serdes.deserialize(v)

    def take(self, n):
        return [(self.key_serdes.deserialize(k), self.value_serdes.deserialize(v)) for k, v in self._take(n)]

    def first(self):
        resp = self.take(n=1)
        if len(resp) < 1:
            raise RuntimeError("table is empty")
        return resp[0]

    def _reduce(self, func: Callable[[V, V], V]):
        raise NotImplementedError(f"{self.__class__.__name__}._reduce")

    def reduce(self, func: Callable[[V, V], V]) -> V:
        return self.value_serdes.deserialize(self._reduce(_lifted_reduce_to_serdes(func, self.value_serdes)))

    def count(self) -> int:
        if self._count_cache is None:
            self._count_cache = self._count()
        return self._count_cache

    def _join(
        self,
        other: "KVTable",
        merge_op: Callable[[V, V], V],
        key_serdes,
        key_serdes_type,
        value_serdes,
        value_serdes_type,
        partitioner,
        partitioner_type,
    ):
        raise NotImplementedError(f"{self.__class__.__name__}._join")

    def join(self, other: "KVTable", merge_op: Callable[[V, V], V] = None):
        return self._join(
            other,
            _lifted_join_merge_to_serdes(merge_op, self.value_serdes),
            key_serdes=self.key_serdes,
            key_serdes_type=self.key_serdes_type,
            value_serdes=self.value_serdes,
            value_serdes_type=self.value_serdes_type,
            partitioner=self.partitioner,
            partitioner_type=self.partitioner_type,
        )

    def _union(
        self,
        other,
        merge_op: Callable[[V, V], V],
        key_serdes,
        key_serdes_type,
        value_serdes,
        value_serdes_type,
        partitioner,
        partitioner_type,
    ):
        raise NotImplementedError(f"{self.__class__.__name__}._union")

    def union(self, other, merge_op: Callable[[V, V], V] = None):
        return self._union(
            other,
            _lifted_join_merge_to_serdes(merge_op, self.value_serdes),
            key_serdes=self.key_serdes,
            key_serdes_type=self.key_serdes_type,
            value_serdes=self.value_serdes,
            value_serdes_type=self.value_serdes_type,
            partitioner=self.partitioner,
            partitioner_type=self.partitioner_type,
        )

    def _subtract_by_key(
        self,
        other: "KVTable",
        key_serdes,
        key_serdes_type,
        value_serdes,
        value_serdes_type,
        partitioner,
        partitioner_type,
    ):
        raise NotImplementedError(f"{self.__class__.__name__}._subtract_by_key")

    def subtractByKey(self, other: "KVTable"):
        return self._subtract_by_key(
            other,
            key_serdes=self.key_serdes,
            key_serdes_type=self.key_serdes_type,
            value_serdes=self.value_serdes,
            value_serdes_type=self.value_serdes_type,
            partitioner=self.partitioner,
            partitioner_type=self.partitioner_type,
        )

    # def save_as(self, name, namespace, partition=None, options=None):
    #     return self.rp.save_as(name=name, namespace=namespace, partition=partition, options=options)


def _serdes_wrapped_generator(_iter, key_serdes, value_serdes):
    for k, v in _iter:
        yield key_serdes.deserialize(k), value_serdes.deserialize(v)


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


def _lifted_join_merge_to_serdes(join_merge_op, value_serdes):
    def _lifted(x, y):
        return value_serdes.serialize(
            join_merge_op(
                value_serdes.deserialize(x),
                value_serdes.deserialize(y),
            )
        )

    return _lifted
