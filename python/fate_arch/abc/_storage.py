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

LOGGER = getLogger()


class StorageTableMetaABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create(self):
        ...

    @abc.abstractmethod
    def set_metas(self, **kwargs):
        ...

    @abc.abstractmethod
    def query_table_meta(self, filter_fields, query_fields=None):
        ...

    @abc.abstractmethod
    def update_metas(self, schema=None, count=None, part_of_data=None, description=None, partitions=None, **kwargs):
        ...

    @abc.abstractmethod
    def destroy_metas(self):
        ...

    @abc.abstractmethod
    def get_name(self):
        ...
        ...

    @abc.abstractmethod
    def get_namespace(self):
        ...

    @abc.abstractmethod
    def get_address(self):
        ...

    @abc.abstractmethod
    def get_engine(self):
        ...

    @abc.abstractmethod
    def get_store_type(self):
        ...

    @abc.abstractmethod
    def get_options(self):
        ...

    @abc.abstractmethod
    def get_partitions(self):
        ...

    @abc.abstractmethod
    def get_in_serialized(self):
        ...

    @abc.abstractmethod
    def get_id_delimiter(self):
        ...

    @abc.abstractmethod
    def get_extend_sid(self):
        ...

    @abc.abstractmethod
    def get_auto_increasing_sid(self):
        ...

    @abc.abstractmethod
    def get_have_head(self):
        ...

    @abc.abstractmethod
    def get_schema(self):
        ...

    @abc.abstractmethod
    def get_count(self):
        ...

    @abc.abstractmethod
    def get_part_of_data(self):
        ...

    @abc.abstractmethod
    def get_description(self):
        ...

    @abc.abstractmethod
    def get_origin(self):
        ...

    @abc.abstractmethod
    def get_disable(self):
        ...

    @abc.abstractmethod
    def to_dict(self) -> dict:
        ...


class StorageTableABC(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self):
        ...

    @property
    @abc.abstractmethod
    def namespace(self):
        ...

    @property
    @abc.abstractmethod
    def address(self):
        ...

    @property
    @abc.abstractmethod
    def engine(self):
        ...

    @property
    @abc.abstractmethod
    def store_type(self):
        ...

    @property
    @abc.abstractmethod
    def options(self):
        ...

    @property
    @abc.abstractmethod
    def partitions(self):
        ...

    @property
    @abc.abstractmethod
    def meta(self) -> StorageTableMetaABC:
        ...

    @meta.setter
    @abc.abstractmethod
    def meta(self, meta: StorageTableMetaABC):
        ...

    @abc.abstractmethod
    def update_meta(self,
                    schema=None,
                    count=None,
                    part_of_data=None,
                    description=None,
                    partitions=None,
                    **kwargs) -> StorageTableMetaABC:
        ...

    @abc.abstractmethod
    def create_meta(self, **kwargs) -> StorageTableMetaABC:
        ...

    @abc.abstractmethod
    def put_all(self, kv_list: Iterable, **kwargs):
        ...

    @abc.abstractmethod
    def collect(self, **kwargs) -> list:
        ...

    @abc.abstractmethod
    def read(self) -> list:
        ...

    @abc.abstractmethod
    def count(self):
        ...

    @abc.abstractmethod
    def destroy(self):
        ...

    @abc.abstractmethod
    def check_address(self):
        ...


class StorageSessionABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def create_table(self, address, name, namespace, partitions, storage_type=None, options=None,
                     **kwargs) -> StorageTableABC:
        ...

    @abc.abstractmethod
    def get_table(self, name, namespace) -> StorageTableABC:
        ...

    @abc.abstractmethod
    def get_table_meta(self, name, namespace) -> StorageTableMetaABC:
        ...

    # @abc.abstractmethod
    # def table(self, name, namespace, address, partitions, store_type=None, options=None, **kwargs) -> StorageTableABC:
    #     ...

    # @abc.abstractmethod
    # def get_storage_info(self, name, namespace):
    #     ...

    @abc.abstractmethod
    def destroy(self):
        ...

    @abc.abstractmethod
    def stop(self):
        ...

    @abc.abstractmethod
    def kill(self):
        ...

    @property
    @abc.abstractmethod
    def session_id(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def engine(self) -> str:
        ...
