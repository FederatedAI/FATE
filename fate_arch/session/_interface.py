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
import typing
from abc import ABCMeta
from collections import Iterable

from fate_arch._interface import GC
from fate_arch.data_table.base import AddressABC
from fate_arch.session._session_types import Party, _FederationParties


# noinspection PyPep8Naming
class TableABC(metaclass=ABCMeta):

    @abc.abstractmethod
    def save(self, name, namespace, **kwargs):
        ...

    @abc.abstractmethod
    def collect(self, **kwargs) -> list:
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
    def map(self, func) -> 'TableABC':
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


class SessionABC(metaclass=ABCMeta):

    @abc.abstractmethod
    def init_federation(self, federation_session_id: str, runtime_conf: dict, **kwargs):
        ...

    @abc.abstractmethod
    def load(self, address: AddressABC, partitions, kwargs) -> TableABC:
        ...

    @abc.abstractmethod
    def parallelize(self, data: Iterable, partition: int, include_key: bool, **kwargs) -> TableABC:
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

    @abc.abstractmethod
    def _get_federation(self):
        ...

    @abc.abstractmethod
    def _get_session_id(self):
        ...

    @abc.abstractmethod
    def _get_federation_parties(self):
        ...

    @property
    def session_id(self) -> str:
        return self._get_session_id()

    @property
    def parties(self) -> '_FederationParties':
        return self._get_federation_parties()

    @property
    def federation(self) -> 'FederationEngineABC':
        return self._get_federation()

    @staticmethod
    def _parse_runtime_conf(runtime_conf):
        role = runtime_conf.get("local").get("role")
        party_id = str(runtime_conf.get("local").get("party_id"))
        party = Party(role, party_id)
        parties = {}
        for role, pid_list in runtime_conf.get("role", {}).items():
            parties[role] = [Party(role, pid) for pid in pid_list]
        return party, parties


class FederationEngineABC(metaclass=ABCMeta):

    @abc.abstractmethod
    def get(self, name: str, tag: str, parties: typing.List[Party], gc: GC) -> typing.List:
        ...

    @abc.abstractmethod
    def remote(self, v, name: str, tag: str, parties: typing.List[Party], gc: GC) -> typing.NoReturn:
        ...
