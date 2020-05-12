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


import abc
from typing import Iterable

import six


# noinspection PyPep8Naming
@six.add_metaclass(abc.ABCMeta)
class Table(object):
    """
    table for distributed computation and storage

    This is a abstract class to be implemented.
    """

    @abc.abstractmethod
    def dtable(self):
        pass

    @classmethod
    def from_dtable(cls, job_id, dtable):
        pass

    @abc.abstractmethod
    def get_partitions(self):
        pass

    @abc.abstractmethod
    def get_namespace(self):
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    @abc.abstractmethod
    def put(self, k, v, use_serialize=True, maybe_large_value=False):
        """
        Stores a key-value record.

        Parameters
        ----------
        k : Key object
          Will be serialized. Must be less than 512 bytes.
        v : object
          Will be serialized. Must be less than 32 MB (or 2G in eggroll 2.x, depends on implements)
        use_serialize : bool, defaults True
        maybe_large_value : bool, defaults False
        Examples
        --------
        >>> from arch.api import session
        >>> a = session.parallelize(range(10))
        >>> a.put('hello', 'world')
        >>> b = a.collect()
        >>> list(b)
        [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), ('hello', 'world')]
        """
        pass

    @abc.abstractmethod
    def put_all(self, kv_list: Iterable, use_serialize=True, chunk_size=100000):
        """
        Puts (key, value) 2-tuple stream from the iterable items.

        Elements must be exact 2-tuples, they may not be of any other type, or tuple subclass.

        Parameters
        ----------
        kv_list : Iterable
          Key-Value 2-tuple iterable. Will be serialized.

        Notes
        -----
        Each key must be less than 512 bytes, value must be less than 32 MB(implementation depends).

        Examples
        --------
        >>> a = session.table('foo', 'bar')
        >>> t = [(1, 2), (3, 4), (5, 6)]
        >>> a.put_all(t)
        >>> list(a.collect())
        [(1, 2), (3, 4), (5, 6)]
        """
        pass

    @abc.abstractmethod
    def put_if_absent(self, k, v, use_serialize=True):
        """
        Stores a key-value record only if the key is not set.

        Parameters
        ----------
        k : key object
          Will be serialized. Must be less than 512 bytes.
        v : Value object
          Will be serialized. Must be less than 32 MB (or 2G in eggroll 2.x, depends on implements)

        Examples
        -------
        >>> a = sessiojn.parallelize(range(10))
        >>> a.put_if_absent(1, 2)
        >>> b = a.collect()
        >>> list(b)
        [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
        >>> a.put_if_absent(-1, -1)
        >>> list(b)
        """
        pass

    @abc.abstractmethod
    def get(self, k, use_serialize=True, maybe_large_value=False):
        """
        Fetches the value matching key.

        Parameters
        ----------
        k : key object
          Will be serialized

        Notes
        -----
        key size Must be less than 512 bytes.

        Returns
        -------
        object
           Corresponding value of the key. Returns None if key does not exist.

        Examples
        --------
        >>> a = session.parallelize(range(10))
        >>> a.get(1)
        (1, 1)
        """
        pass

    @abc.abstractmethod
    def collect(self, min_chunk_size=0, use_serialize=True) -> list:
        """
        Returns an iterator of (key, value) 2-tuple from the Table.

        Parameters
        ---------
        min_chunk_size : int
          Minimum chunk size (key bytes + value bytes) returned if end of table is not hit.
          0 indicates a default chunk size (partition_num * 1.75 MB)
          negative number indicates no chunk limit, i.e. returning all records.
          Default chunk size is recommended if there is no special needs from user.

        Returns
        -------
        Iterator

        Examples
        --------
        >>> a = session.parallelize(range(10))
        >>> b = a.collect(min_chunk_size=1000)
        >>>list(b)
        [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
        """
        pass

    @abc.abstractmethod
    def delete(self, k, use_serialize=True):
        """
        Returns the deleted value corresponding to the key.

        Parameters
        ----------
          k : object
            Key object. Will be serialized. Must be less than 512 bytes.
        Returns
        -------
        object
          Corresponding value of the deleted key. Returns None if key does not exist.

        Examples
        --------
        >>> a = session.parallelize(range(10))
        >>> a.delete(1)
        1
        """
        pass

    @abc.abstractmethod
    def destroy(self):
        """
        Destroys this Table, freeing its associated storage resources.

        Returns
        -------
        None

        Examples
        ----------
        >>> a = session.parallelize(range(10))
        >>> a.destroy()
        """
        pass

    @abc.abstractmethod
    def count(self):
        """
        Returns the number of elements in the Table.

        Returns
        -------
        int
          Number of elements in this Table.

        Examples
        --------
        >>> a = session.parallelize(range(10))
        >>> a.count()
        10
        """
        pass

    @abc.abstractmethod
    def take(self, n=1, keysOnly=False, use_serialize=True):
        """
        Returns the first n element(s) of a Table.

        Parameters
        ----------
        n : int
          Number of top data returned.
        keysOnly : bool
          Whether to return keys only. `True` returns keys only and `False` returns both keys and values.

        Returns
        -------
        list
          Lists of top n keys or key-value pairs.

        Examples
        --------
        >>> a = session.parallelize([1, 2, 3])
        >>> a.take(2)
        [(1, 1), (2, 2)]
        >>> a.take(2, keysOnly=True)
        [1, 2]
        """
        pass

    @abc.abstractmethod
    def first(self, keysOnly=False, use_serialize=True):
        """
        Returns the first element of a Table. Shortcut of `take(1, keysOnly)`

        Parameters
        ----------
        keysOnly : bool
          Whether to return keys only. `True` returns keys only and `False` returns both keys and values.
        use_serialize : bool

        Returns
        -------
        tuple or object
          First element of the Table. It is a tuple if `keysOnly=False`, or an object if `keysOnly=True`.

        Examples
        --------
        >>> a = session.parallelize([1, 2, 3])
        >>> a.first()
        (1, 1)
        """
        pass

    @abc.abstractmethod
    def save_as(self, name, namespace, partition=None, use_serialize=True, **kwargs):
        """
        Transforms a temporary table to a persistent table.

        Parameters
        ----------
        name : string
          Table name of result Table.
        namespace: string
          Table namespace of result Table.
        partition : int
          Number of partition for the new persistent table.
        use_serialize

        Returns
        -------
        Table
           Result persistent Table.

        Examples
        --------
        >>> a = session.parallelize(range(10))
        >>> b = a.save_as('foo', 'bar', partition=2)
        """
        pass

    """
    computing apis
    """

    @abc.abstractmethod
    def map(self, func):
        """
        Returns a new Table by applying a function to each (key, value) 2-tuple of this Table.

        Notes
        -----
        In-place computing does not apply.

        Parameters
        ----------
        func : k1, v1 -> k2, v2
          The function applying to each 2-tuple.

        Returns
        -------
        Table
          A new table containing results.

        Examples
        --------
        >>> a = session.parallelize(['a', 'b', 'c'])    # [(0, 'a'), (1, 'b'), (2, 'c')]
        >>> b = a.map(lambda k, v: (v, v + '1'))
        >>> list(b.collect())
        [("a", "a1"), ("b", "b1"), ("c", "c1")]
        """
        pass

    @abc.abstractmethod
    def mapValues(self, func):
        """
        Returns a Table by applying a function to each value of this Table, while keys do not change.

        Parameters
        ----------
        func : v1 -> v2
          The function applying to each value.

        Returns
        -------
        Table
          A new table containing results.

        Examples
        --------
        >>> from arch.api import session
        >>> a = session.parallelize([('a', ['apple', 'banana', 'lemon']), ('b', ['grapes'])], include_key=True)
        >>> b = a.mapValues(lambda x: len(x))
        >>> list(b.collect())
        [('a', 3), ('b', 1)]
        """
        pass

    @abc.abstractmethod
    def mapPartitions(self, func):
        """
        Returns a new Table by applying a function to each partition of this Table.

        Parameters
        ----------
        func : iter -> v
          The function applying to each partition.

        Returns
        -------
        Table
          A new table with k-v: uuid key - v.

        Examples
        --------
        >>> a = session.parallelize([1, 2, 3, 4, 5], partition=2)
        >>> def f(iterator):
        >>> 	sum = 0
        >>> 	for k, v in iterator:
        >>> 		sum += v
        >>> 	return sum
        >>> b = a.mapPartitions(f)
        >>> list(b.collect())
        [(3, 6), (4, 9)]
        """
        pass

    def mapPartitions2(self, func):
        """
        Returns a new Table by applying a function to each partition of this Table.

        Notes
        -----
        In-place computing does not apply.

        Parameters
        ----------
        func : iter -> (k, v)
          The function applying to each partition.

        Returns
        -------
        Table
          A new table containing results.

        Examples
        --------
        >>> from arch.api import session
        >>> a = session.parallelize([1, 2, 3, 4, 5], partition=2)
        >>> def f(iterator):
        ...     s = 0
        ... 	for k, v in iterator:
        ... 		s += v
        ...	    return [(s, s)]
        ...

        >>> b = a.mapPartitions2(f)
        >>> list(b.collect())
        [(6, 6), (9, 9)]
        """
        pass

    @abc.abstractmethod
    def flatMap(self, func):
        """
        Returns a new Table by first applying func, then flattening it.

        Notes
        ----
        In-place computing does not apply.

        Parameters
        ----------
        func : k, v -> list
          The function applying to each 2-tuple.

        Returns
        -------
        list
          A list containing all flattened elements within each list after applying `func`.
          'Last-Write Win' policy is used if key exists in multiple lists.

        Examples
        --------
        >>> import random
        >>> def foo(k, v):
                result = []
                r = random.randint(10000, 99999)
                for i in range(0, k):
                    result.append((k + r + i, v + r + i))
                return result
        >>> a = session.parallelize(range(5))
        >>> b = a.flatMap(foo)
        >>> list(b.collect())
        [(83030, 83030), (84321, 84321), (84322, 84322), (91266, 91266), (91267, 91267), (91268, 91268), (91269, 91269), (71349, 71349), (71350, 71350), (71351, 71351)]
        """
        pass

    @abc.abstractmethod
    def reduce(self, func, key_func=None):
        """
        Reduces the elements of this Table using the specified associative binary operator.

        Currently reduces partitions locally.

        Parameters
        ----------
        func : v1, v2 -> v
          Binary operator applying to each 2-tuple.
        key_func : k -> k'
          Unary operator applying to each key to obtain the real key for reducing.
          Defaults to None, which means reducing on original key.

        Returns
        -------
        Table
          A new table containing results.

        Examples
        --------
        >>> from operator import add
        >>> session.parallelize([1, 2, 3, 4, 5]).reduce(add)
        >>> 15
        """
        pass

    @abc.abstractmethod
    def join(self, other, func):
        """
        Returns a Table containing all pairs of elements with matching keys in self and other, i.e. 'inner join'.

        Each pair of elements will be returned as a (k, func(v1, v2)) tuple,
        where (k, v1) is in self and (k, v2) in other.

        Notes
        -----
        In-place computing applies if enabled. Results will be in left Table (the caller).

        Parameters
        ----------
        other : Table
          Another Table to be operated with.
        func : v1, v2 -> v
          Binary operator applying to values whose key exists in both Tables.

        Returns
        -------
        Table
          A Table containing results.

        Examples
        --------
        >>> a = session.parallelize([('a', 1), ('b', 4)], include_key=True)
        >>> b = session.parallelize([('a', 2), ('c', 3)], include_key=True)
        >>> c = a.join(b, lambda v1, v2: v1 + v2)
        >>> list(c.collect())
        [('a', 3)]
        """
        pass

    @abc.abstractmethod
    def glom(self):
        """
        Coalesces all elements within each partition into a list.

        Returns
        -------
        list
          A list containing all coalesced partition and its elements.
          First element of each tuple is chosen from key of last element of each partition.

        Examples
        --------
        >>> a = session.parallelize(range(5), partition=3).glom().collect()
        >>> list(a)
        [(2, [(2, 2)]), (3, [(0, 0), (3, 3)]), (4, [(1, 1), (4, 4)])]
        """
        pass

    @abc.abstractmethod
    def sample(self, fraction, seed=None):
        """
        Return a sampled subset of this Table.

        Notes
        ----
        In-place computing does not apply.

        Parameters
        ----------
        fraction : float
          Expected size of the sample as a fraction of this Table's size
          without replacement: probability that each element is chosen.
          Fraction must be [0, 1] with replacement: expected number of times each element is chosen.
        seed : int
          Seed of the random number generator. Use current timestamp when `None` is passed.

        Returns
        -------
        Table
          A new table containing results.

        Examples
        --------
        >>> x = session.parallelize(range(100), partition=4)
        >>>  6 <= x.sample(0.1, 81).count() <= 14
        True
        """
        pass

    @abc.abstractmethod
    def subtractByKey(self, other):
        """
        Returns a new Table containing elements only in this Table but not in the other Table.

        Parameters
        ----------
        other : Table
          Another Table to be operated with.

        Notes
        -----
        In-place computing applies if enabled. Result will be in left Table (the caller).

        Returns
        -------
        Table
          A new table containing results.

        Examples
        --------
        >>> a = session.parallelize(range(10))
        >>> b = session.parallelize(range(5))
        >>> c = a.subtractByKey(b)
        >>> list(c.collect())
        [(5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
        """
        pass

    @abc.abstractmethod
    def filter(self, func):
        """
        Returns a new Table containing only those keys which satisfy a predicate passed in via `func`.

        Parameters
        ----------
        func : k, v -> bool
          Predicate function returning a boolean.
        Notes
        ----
        In-place computing does not apply.

        Returns
        -------
        Table
          A new table containing results.

        Examples
        --------
        >>> a = session.parallelize([0, 1, 2])
        >>> b = a.filter(lambda k, v : k % 2 == 0)
        >>> list(b.collect())
        [(0, 0), (2, 2)]
        >>> c = a.filter(lambda k, v : v % 2 != 0)
        >>> list(c.collect())
        [(1, 1)]
        """
        pass

    @abc.abstractmethod
    def union(self, other, func=lambda v1, v2: v1):
        """
        Returns union of this Table and the other Table.

        Function will be applied to values of keys that exist in both table.

        Notes
        -----
        In-place computing applies if enabled. Result will be in left Table (the caller).

        Parameters
        ----------
        other : Table
          Another Table to be operated with.
        func : v1, v2 -> v
          The function applying to values whose key exists in both Tables. Default using left table's value.

        Returns
        -------
        Table
          A table containing results.

        Examples
        --------
        >>> a = session.parallelize([1, 2, 3])	# [(0, 1), (1, 2), (2, 3)]
        >>> b = session.parallelize([(1, 1), (2, 2), (3, 3)])
        >>> c = a.union(b, lambda v1, v2 : v1 + v2)
        >>> list(c.collect())
        [(0, 1), (1, 3), (2, 5), (3, 3)]
        """
        pass


    """
    meta utils
    """

    def get_meta(self, key):
        from .session import FateSession
        return FateSession.get_data_table_meta(key,
                                               data_table_name=self.get_name(),
                                               data_table_namespace=self.get_namespace())

    def get_metas(self):
        from .session import FateSession
        return FateSession.get_data_table_metas(data_table_name=self.get_name(),
                                                data_table_namespace=self.get_namespace())

    def save_metas(self, kv):
        from .session import FateSession
        return FateSession.save_data_table_meta(kv=kv,
                                                data_table_name=self.get_name(),
                                                data_table_namespace=self.get_namespace())
