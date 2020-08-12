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
from fate_arch.common.log import getLogger
from fate_arch.common import StorageTableMetaType
from fate_arch.computing import ComputingType
from fate_arch.abc import AddressABC

MAX_NUM = 10000

LOGGER = getLogger()


class StorageTableABC(metaclass=abc.ABCMeta):
    """
    table for distributed storage
    """

    @abc.abstractmethod
    def get_partitions(self):
        pass

    @abc.abstractmethod
    def get_name(self):
        pass

    @abc.abstractmethod
    def get_namespace(self):
        pass

    @abc.abstractmethod
    def get_storage_engine(self):
        pass

    @abc.abstractmethod
    def get_address(self):
        pass

    @abc.abstractmethod
    def put_all(self, kv_list: Iterable, **kwargs):
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
        """
        pass

    @abc.abstractmethod
    def collect(self, **kwargs) -> list:
        """
        Returns an iterator of (key, value) 2-tuple from the Table.

        Returns
        -------
        Iterator
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
        """
        pass

    @abc.abstractmethod
    def save_as(self, name, namespace, partition=None, schema=None, **kwargs):
        pass

    @abc.abstractmethod
    def close(self):
        pass

    @abc.abstractmethod
    def destroy(self):
        ...

    @abc.abstractmethod
    def get_meta(self, _type=StorageTableMetaType.SCHEMA, name=None, namespace=None):
        ...

    @abc.abstractmethod
    def save_meta(self, schema=None, name=None, namespace=None, party_of_data=None, count=0, partitions=1):
        ...

    @abc.abstractmethod
    def destroy_meta(self):
        ...


class StorageSessionABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create_table(self, address, name, namespace, partitions, storage_type=None, options=None, **kwargs) -> StorageTableABC:
        pass

    @abc.abstractmethod
    def get_table(self, name, namespace) -> StorageTableABC:
        pass

    @abc.abstractmethod
    def get_storage_info(self, name, namespace):
        pass

    @abc.abstractmethod
    def get_address(self, storage_engine, address_dict) -> AddressABC:
        pass

    @abc.abstractmethod
    def convert(self, src_table, dest_name, dest_namespace, session_id, computing_engine: ComputingType, force=False, **kwargs):
        pass

    @abc.abstractmethod
    def copy_table(self, src_table, dest_table):
        pass

    @abc.abstractmethod
    def stop(self):
        pass

    @abc.abstractmethod
    def kill(self):
        pass
