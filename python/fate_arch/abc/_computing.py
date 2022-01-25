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

"""
distributed computing
"""

import abc
import typing
from abc import ABCMeta
from collections import Iterable

from fate_arch.abc._address import AddressABC
from fate_arch.abc._path import PathABC

__all__ = ["CTableABC", "CSessionABC"]


# noinspection PyPep8Naming
class CTableABC(metaclass=ABCMeta):
    """
    a table of pair-like data supports distributed processing
    """

    @property
    @abc.abstractmethod
    def engine(self):
        """
        get the engine name of table

        Returns
        -------
        int
           number of partitions
        """
        ...

    @property
    @abc.abstractmethod
    def partitions(self):
        """
        get the partitions of table

        Returns
        -------
        int
           number of partitions
        """
        ...

    @abc.abstractmethod
    def copy(self):
        ...

    @abc.abstractmethod
    def save(self, address: AddressABC, partitions: int, schema: dict, **kwargs):
        """
        save table

        Parameters
        ----------
        address: AddressABC
           address to save table to
        partitions: int
           number of partitions to save as
        schema: dict
           table schema
        """
        ...

    @abc.abstractmethod
    def collect(self, **kwargs) -> typing.Generator:
        """
        collect data from table

        Returns
        -------
        generator
           generator of data

        Notes
        ------
        no order guarantee
        """
        ...

    @abc.abstractmethod
    def take(self, n=1, **kwargs):
        """
        take ``n`` data from table

        Parameters
        ----------
        n: int
          number of data to take

        Returns
        -------
        list
           a list of ``n`` data

        Notes
        ------
        no order guarantee
        """
        ...

    @abc.abstractmethod
    def first(self, **kwargs):
        """
        take one data from table

        Returns
        -------
        object
          a data from table


        Notes
        -------
        no order guarantee
        """
        ...

    @abc.abstractmethod
    def count(self) -> int:
        """
        number of data in table

        Returns
        -------
        int
           number of data
        """
        ...

    @abc.abstractmethod
    def map(self, func) -> 'CTableABC':
        """
        apply `func` to each data

        Parameters
        ----------
        func: ``typing.Callable[[object, object], typing.Tuple[object, object]]``
           function map (k1, v1) to (k2, v2)

        Returns
        -------
        CTableABC
           A new table

        Examples
        --------
        >>> from fate_arch.session import computing_session
        >>> a = computing_session.parallelize([('k1', 1), ('k2', 2), ('k3', 3)], include_key=True, partition=2)
        >>> b = a.map(lambda k, v: (k, v**2))
        >>> list(b.collect())
        [("k1", 1), ("k2", 4), ("k3", 9)]
        """
        ...

    @abc.abstractmethod
    def mapValues(self, func):
        """
        apply `func` to each value of data

        Parameters
        ----------
        func: ``typing.Callable[[object], object]``
           map v1 to v2

        Returns
        -------
        CTableABC
           A new table

        Examples
        --------
        >>> from fate_arch.session import computing_session
        >>> a = computing_session.parallelize([('a', ['apple', 'banana', 'lemon']), ('b', ['grapes'])], include_key=True, partition=2)
        >>> b = a.mapValues(lambda x: len(x))
        >>> list(b.collect())
        [('a', 3), ('b', 1)]
        """
        ...

    @abc.abstractmethod
    def mapPartitions(self, func, use_previous_behavior=True, preserves_partitioning=False):
        """
        apply ``func`` to each partition of table

        Parameters
        ----------
        func: ``typing.Callable[[iter], list]``
           accept an iterator of pair, return a list of pair
        use_previous_behavior: bool
           this parameter is provided for compatible reason, if set True, call this func will call ``applyPartitions`` instead
        preserves_partitioning: bool
           flag indicate whether the `func` will preserve partition

        Returns
        -------
        CTableABC
           a new table

        Examples
        --------
        >>> from fate_arch.session import computing_session
        >>> a = computing_session.parallelize([1, 2, 3, 4, 5], include_key=False, partition=2)
        >>> def f(iterator):
        ...     s = 0
        ... 	for k, v in iterator:
        ... 		s += v
        ...	    return [(s, s)]
        ...
        >>> b = a.mapPartitions(f)
        >>> list(b.collect())
        [(6, 6), (9, 9)]
        """
        ...

    @abc.abstractmethod
    def mapReducePartitions(self, mapper, reducer, **kwargs):
        """
        apply ``mapper`` to each partition of table and then perform reduce by key operation with `reducer`

        Parameters
        ----------
        mapper: ``typing.Callable[[iter], list]``
           accept an iterator of pair, return a list of pair
        reducer: ``typing.Callable[[object, object], object]``
           reduce v1, v2 to v3

        Returns
        -------
        CTableABC
           a new table

        Examples
        --------
        >>> from fate_arch.session import computing_session
        >>> table = computing_session.parallelize([(1, 2), (2, 3), (3, 4), (4, 5)], include_key=False, partition=2)
        >>> def _mapper(it):
        ...     r = []
        ...     for k, v in it:
        ...         r.append((k % 3, v**2))
        ...         r.append((k % 2, v ** 3))
        ...     return r
        >>> def _reducer(a, b):
        ...     return a + b
        >>> output = table.mapReducePartitions(_mapper, _reducer)
        >>> collected = dict(output.collect())
        >>> assert collected[0] == 3 ** 3 + 5 ** 3 + 4 ** 2
        >>> assert collected[1] == 2 ** 3 + 4 ** 3 + 2 ** 2 + 5 ** 2
        >>> assert collected[2] == 3 ** 2
        """

        ...

    def applyPartitions(self, func):
        """
        apply ``func`` to each partitions as a single object

        Parameters
        ----------
        func: ``typing.Callable[[iter], object]``
           accept a iterator, return a object

        Returns
        -------
        CTableABC
           a new table, with each partition contains a single key-value pair

        Examples
        --------
        >>> from fate_arch.session import computing_session
        >>> a = computing_session.parallelize([1, 2, 3], partition=3, include_key=False)
        >>> def f(it):
        ...    r = []
        ...    for k, v in it:
        ...        r.append(v, v**2, v**3)
        ...    return r
        >>> output = a.applyPartitions(f)
        >>> assert (2, 2**2, 2**3) in [v[0] for _, v in output.collect()]
        """
        ...

    @abc.abstractmethod
    def flatMap(self, func):
        """
        apply a flat ``func`` to each data of table

        Parameters
        ----------
        func: ``typing.Callable[[object, object], typing.List[object, object]]``
           a flat function accept two parameters return a list of pair

        Returns
        -------
        CTableABC
           a new table

        Examples
        --------
        >>> from fate_arch.session import computing_session
        >>> a = computing_session.parallelize([(1, 1), (2, 2)], include_key=True, partition=2)
        >>> b = a.flatMap(lambda x, y: [(x, y), (x + 10, y ** 2)])
        >>> c = list(b.collect())
        >>> assert len(c) = 4
        >>> assert ((1, 1) in c) and ((2, 2) in c) and ((11, 1) in c) and ((12, 4) in c)
        """
        ...

    @abc.abstractmethod
    def reduce(self, func):
        """
        reduces all value in pair of table by a binary function `func`

        Parameters
        ----------
        func: typing.Callable[[object, object], object]
           binary function reduce two value into one

        Returns
        -------
        object
           a single object



        Examples
        --------
        >>> from fate_arch.session import computing_session
        >>> a = computing_session.parallelize(range(100), include_key=False, partition=4)
        >>> assert a.reduce(lambda x, y: x + y) == sum(range(100))

        Notes
        ------
        `func` should be associative
        """
        ...

    @abc.abstractmethod
    def glom(self):
        """
        coalesces all data within partition into a list

        Returns
        -------
        list
           list containing all coalesced partition and its elements.
           First element of each tuple is chosen from key of last element of each partition.

        Examples
        --------
        >>> from fate_arch.session import computing_session
        >>> a = computing_session.parallelize(range(5), include_key=False, partition=3).glom().collect()
        >>> list(a)
        [(2, [(2, 2)]), (3, [(0, 0), (3, 3)]), (4, [(1, 1), (4, 4)])]
        """
        ...

    @abc.abstractmethod
    def sample(self, *, fraction: typing.Optional[float] = None, num: typing.Optional[int] = None, seed=None):
        """
        return a sampled subset of this Table.
        Parameters
        ----------
        fraction: float
          Expected size of the sample as a fraction of this table's size
          without replacement: probability that each element is chosen.
          Fraction must be [0, 1] with replacement: expected number of times each element is chosen.
        num: int
          Exact number of the sample from this table's size
        seed: int
          Seed of the random number generator. Use current timestamp when `None` is passed.

        Returns
        -------
        CTableABC
           a new table

        Examples
        --------
        >>> from fate_arch.session import computing_session
        >>> x = computing_session.parallelize(range(100), include_key=False, partition=4)
        >>> 6 <= x.sample(fraction=0.1, seed=81).count() <= 14
        True

        Notes
        -------
        use one of ``fraction`` and ``num``, not both

        """
        ...

    @abc.abstractmethod
    def filter(self, func):
        """
        returns a new table containing only those keys which satisfy a predicate passed in via ``func``.

        Parameters
        ----------
        func: typing.Callable[[object, object], bool]
           Predicate function returning a boolean.

        Returns
        -------
        CTableABC
           A new table containing results.

        Examples
        --------
        >>> from fate_arch.session import computing_session
        >>> a = computing_session.parallelize([0, 1, 2], include_key=False, partition=2)
        >>> b = a.filter(lambda k, v : k % 2 == 0)
        >>> list(b.collect())
        [(0, 0), (2, 2)]
        >>> c = a.filter(lambda k, v : v % 2 != 0)
        >>> list(c.collect())
        [(1, 1)]
        """
        ...

    @abc.abstractmethod
    def join(self, other, func):
        """
        returns intersection of this table and the other table.

        function ``func`` will be applied to values of keys that exist in both table.

        Parameters
        ----------
        other: CTableABC
          another table to be operated with.
        func: ``typing.Callable[[object, object], object]``
          the function applying to values whose key exists in both tables.
          default using left table's value.

        Returns
        -------
        CTableABC
          a new table

        Examples
        --------
        >>> from fate_arch.session import computing_session
        >>> a = computing_session.parallelize([1, 2, 3], include_key=False, partition=2)	# [(0, 1), (1, 2), (2, 3)]
        >>> b = computing_session.parallelize([(1, 1), (2, 2), (3, 3)], include_key=True, partition=2)
        >>> c = a.join(b, lambda v1, v2 : v1 + v2)
        >>> list(c.collect())
        [(1, 3), (2, 5)]
        """
        ...

    @abc.abstractmethod
    def union(self, other, func=lambda v1, v2: v1):
        """
        returns union of this table and the other table.

        function ``func`` will be applied to values of keys that exist in both table.

        Parameters
        ----------
        other: CTableABC
          another table to be operated with.
        func: ``typing.Callable[[object, object], object]``
          The function applying to values whose key exists in both tables.
          default using left table's value.

        Returns
        -------
        CTableABC
          a new table

        Examples
        --------
        >>> from fate_arch.session import computing_session
        >>> a = computing_session.parallelize([1, 2, 3], include_key=False, partition=2)	# [(0, 1), (1, 2), (2, 3)]
        >>> b = computing_session.parallelize([(1, 1), (2, 2), (3, 3)], include_key=True, partition=2)
        >>> c = a.union(b, lambda v1, v2 : v1 + v2)
        >>> list(c.collect())
        [(0, 1), (1, 3), (2, 5), (3, 3)]
        """
        ...

    @abc.abstractmethod
    def subtractByKey(self, other):
        """
        returns a new table containing elements only in this table but not in the other table.

        Parameters
        ----------
        other: CTableABC
          Another table to be subtractbykey with.

        Returns
        -------
        CTableABC
          A new table

        Examples
        --------
        >>> from fate_arch.session import computing_session
        >>> a = computing_session.parallelize(range(10), include_key=False, partition=2)
        >>> b = computing_session.parallelize(range(5), include_key=False, partition=2)
        >>> c = a.subtractByKey(b)
        >>> list(c.collect())
        [(5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
        """
        ...

    @property
    def schema(self):
        if not hasattr(self, "_schema"):
            setattr(self, "_schema", {})
        return getattr(self, "_schema")

    @schema.setter
    def schema(self, value):
        setattr(self, "_schema", value)


class CSessionABC(metaclass=ABCMeta):
    """
    computing session to load/create/clean tables
    """

    @abc.abstractmethod
    def load(self, address: AddressABC, partitions, schema: dict, **kwargs) -> typing.Union[PathABC, CTableABC]:
        """
        load a table from given address

        Parameters
        ----------
        address: AddressABC
           address to load table from
        partitions: int
           number of partitions of loaded table
        schema: dict
           schema associate with this table

        Returns
        -------
        CTableABC
           a table in memory
        """
        ...

    @abc.abstractmethod
    def parallelize(self, data: Iterable, partition: int, include_key: bool, **kwargs) -> CTableABC:
        """
        create table from iterable data

        Parameters
        ----------
        data: Iterable
           data to create table from
        partition: int
           number of partitions of created table
        include_key: bool
           ``True`` for create table directly from data, ``False`` for create table with generated keys start from 0

        Returns
        -------
        CTableABC
           a table create from data

        """
        pass

    @abc.abstractmethod
    def cleanup(self, name, namespace):
        """
        delete table(s)

        Parameters
        ----------
        name: str
           table name or wildcard character
        namespace: str
           namespace
        """

    @abc.abstractmethod
    def stop(self):
        pass

    @abc.abstractmethod
    def kill(self):
        pass

    @property
    @abc.abstractmethod
    def session_id(self) -> str:
        """
        get computing session id

        Returns
        -------
        str
           computing session id
        """
        ...
