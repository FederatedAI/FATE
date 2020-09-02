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


# noinspection PyPep8Naming
class CTableABC(metaclass=ABCMeta):

    @property
    @abc.abstractmethod
    def partitions(self):
        ...

    @abc.abstractmethod
    def save(self, address: AddressABC, partitions: int, schema: dict, **kwargs):
        ...

    @abc.abstractmethod
    def collect(self, **kwargs) -> typing.Generator:
        ...

    @abc.abstractmethod
    def take(self, n=1, **kwargs):
        ...

    @abc.abstractmethod
    def first(self, **kwargs):
        ...

    @abc.abstractmethod
    def count(self) -> int:
        ...

    @abc.abstractmethod
    def map(self, func) -> 'CTableABC':
        ...

    @abc.abstractmethod
    def mapValues(self, func):
        ...

    @abc.abstractmethod
    def mapPartitions(self, func):
        ...

    @abc.abstractmethod
    def flatMap(self, func):
        ...

    @abc.abstractmethod
    def reduce(self, func):
        ...

    @abc.abstractmethod
    def glom(self):
        ...

    @abc.abstractmethod
    def sample(self, fraction, seed=None):
        ...

    @abc.abstractmethod
    def filter(self, func):
        ...

    @abc.abstractmethod
    def join(self, other, func):
        ...

    @abc.abstractmethod
    def union(self, other, func=lambda v1, v2: v1):
        ...

    @abc.abstractmethod
    def subtractByKey(self, other):
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

    @abc.abstractmethod
    def load(self, address: AddressABC, partitions, schema: dict, **kwargs) -> typing.Union[PathABC, CTableABC]:
        ...

    @abc.abstractmethod
    def parallelize(self, data: Iterable, partition: int, include_key: bool, **kwargs) -> CTableABC:
        """
        create table from iterable data
        """
        pass

    @abc.abstractmethod
    def cleanup(self, name, namespace):
        """
        delete table(s)
        Parameters
        ----------
        name: table name or wildcard character
        namespace: namespace
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
        ...
