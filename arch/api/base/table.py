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

    This is a abstract class to be impl.
    """

    @abc.abstractmethod
    def get_namespace(self):
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    @abc.abstractmethod
    def dtable(self):
        pass

    @classmethod
    def from_dtable(cls, job_id, dtable):
        pass

    @abc.abstractmethod
    def save_as(self, name, namespace, partition=None, use_serialize=True):
        pass

    @abc.abstractmethod
    def put(self, k, v, use_serialize=True, maybe_large_value=False):
        pass

    @abc.abstractmethod
    def put_all(self, kv_list: Iterable, use_serialize=True, chunk_size=100000):
        pass

    @abc.abstractmethod
    def get(self, k, use_serialize=True, maybe_large_value=False):
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
    def put_if_absent(self, k, v, use_serialize=True):
        pass

    @abc.abstractmethod
    def take(self, n=1, keysOnly=False, use_serialize=True):
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
    def get_partitions(self):
        pass

    """
    computing apis
    """

    @abc.abstractmethod
    def map(self, func):
        """
        Args:
            func:
        """
        pass

    @abc.abstractmethod
    def mapValues(self, func):
        pass

    @abc.abstractmethod
    def mapPartitions(self, func):
        pass

    def mapPartitions2(self, func):
        pass

    @abc.abstractmethod
    def reduce(self, func):
        pass

    @abc.abstractmethod
    def join(self, other, func):
        pass

    @abc.abstractmethod
    def glom(self):
        pass

    @abc.abstractmethod
    def sample(self, fraction, seed=None):
        pass

    @abc.abstractmethod
    def subtractByKey(self, other):
        pass

    @abc.abstractmethod
    def filter(self, func):
        pass

    @abc.abstractmethod
    def union(self, other, func=lambda v1, v2: v1):
        pass

    @abc.abstractmethod
    def flatMap(self, func):
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
                                                data_table_name="%s.meta" % self.get_name(),
                                                data_table_namespace=self.get_namespace())
