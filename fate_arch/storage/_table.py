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


import operator
from typing import Iterable

import peewee

from fate_arch.abc import StorageTableMetaABC, StorageTableABC, AddressABC
from fate_arch.common.base_utils import current_timestamp
from fate_arch.common.log import getLogger
from fate_arch.storage import Relationship
from fate_arch.storage.metastore.db_models import DB, StorageTableMetaModel

MAX_NUM = 10000

LOGGER = getLogger()


class StorageTableBase(StorageTableABC):
    def __init__(self, name, namespace):
        self._name = name
        self._namespace = namespace
        self._meta = None

    def destroy(self):
        # destroy schema
        self._meta.destroy_metas()
        # subclass method needs do: super().destroy()

    def set_meta(self, meta):
        self._meta = meta

    def get_meta(self):
        return self._meta

    def get_name(self):
        pass

    def get_namespace(self):
        pass

    def get_address(self):
        pass

    def get_engine(self):
        pass

    def get_type(self):
        pass

    def get_options(self):
        pass

    def get_partitions(self):
        pass

    def put_all(self, kv_list: Iterable, **kwargs):
        pass

    def collect(self, **kwargs) -> list:
        pass

    def count(self):
        pass

    def save_as(self, dest_name, dest_namespace, partitions=None, schema=None):
        src_table_meta = self.get_meta()
        pass


class StorageTableMeta(StorageTableMetaABC):
    def __init__(self, name, namespace):
        self.name = name
        self.namespace = namespace
        self.address = None
        self.engine = None
        self.type = None
        self.options = None
        self.partitions = None
        self.schema = None
        self.count = None
        self.part_of_data = None
        self.description = None
        self.create_time = None
        self.update_time = None
        self.build()

    def build(self):
        for k, v in self.table_meta.__dict__["__data__"].items():
            setattr(self, k.lstrip("f_"), v)
        self.address = self.create_address(storage_engine=self.engine, address_dict=self.address)

    def __new__(cls, *args, **kwargs):
        name, namespace = kwargs.get("name"), kwargs.get("namespace")
        if not name or not namespace:
            return None
        tables_meta = cls.query_table_meta(filter_fields=dict(name=name, namespace=namespace))
        if not tables_meta:
            return None
        self = super().__new__(cls)
        setattr(self, "table_meta", tables_meta[0])
        return self

    @classmethod
    def create_metas(cls, **kwargs):
        with DB.connection_context():
            table_meta = StorageTableMetaModel()
            table_meta.f_create_time = current_timestamp()
            table_meta.f_schema = {}
            table_meta.f_part_of_data = {}
            for k, v in kwargs.items():
                attr_name = 'f_%s' % k
                if hasattr(StorageTableMetaModel, attr_name):
                    setattr(table_meta, attr_name, v if not issubclass(type(v), AddressABC) else v.__dict__)
            try:
                rows = table_meta.save(force_insert=True)
                if rows != 1:
                    raise Exception("create table meta failed")
            except peewee.IntegrityError as e:
                if e.args[0] == 1062:
                    # warning
                    pass
                else:
                    raise e
            except Exception as e:
                raise e

    @classmethod
    def query_table_meta(cls, filter_fields, query_fields=None):
        with DB.connection_context():
            filters = []
            querys = []
            for f_n, f_v in filter_fields.items():
                attr_name = 'f_%s' % f_n
                if hasattr(StorageTableMetaModel, attr_name):
                    filters.append(operator.attrgetter('f_%s' % f_n)(StorageTableMetaModel) == f_v)
            if query_fields:
                for f_n in query_fields:
                    attr_name = 'f_%s' % f_n
                    if hasattr(StorageTableMetaModel, attr_name):
                        querys.append(operator.attrgetter('f_%s' % f_n)(StorageTableMetaModel))
            if filters:
                if querys:
                    tables_meta = StorageTableMetaModel.select(querys).where(*filters)
                else:
                    tables_meta = StorageTableMetaModel.select().where(*filters)
                return [table_meta for table_meta in tables_meta]
            else:
                # not allow query all table
                return []

    def update_metas(self, schema=None, count=None, part_of_data=None, description=None, partitions=None, **kwargs):
        meta_info = {}
        for k, v in locals().items():
            if k not in ["self", "kwargs", "meta_info"] and v is not None:
                meta_info[k] = v
        meta_info.update(kwargs)
        meta_info["name"] = meta_info.get("name", self.name)
        meta_info["namespace"] = meta_info.get("namespace", self.namespace)
        with DB.connection_context():
            update_filters = []
            primary_keys = StorageTableMetaModel._meta.primary_key.field_names
            for p_k in primary_keys:
                update_filters.append(operator.attrgetter(p_k)(StorageTableMetaModel) == meta_info[p_k.lstrip("f_")])
            table_meta = StorageTableMetaModel()
            update_fields = {}
            for k, v in meta_info.items():
                attr_name = 'f_%s' % k
                if hasattr(StorageTableMetaModel, attr_name) and attr_name not in primary_keys:
                    if k == "part_of_data":
                        if len(v) < 100:
                            tmp = table_meta.f_part_of_data[- (100 - len(v)):] + v
                        else:
                            tmp = v[:100]
                        update_fields[operator.attrgetter(attr_name)(StorageTableMetaModel)] = tmp
                    else:
                        update_fields[operator.attrgetter(attr_name)(StorageTableMetaModel)] = v
            if update_filters:
                operate = table_meta.update(update_fields).where(*update_filters)
            else:
                operate = table_meta.update(update_fields)
            return operate.execute() > 0

    def destroy_metas(self):
        with DB.connection_context():
            StorageTableMetaModel \
                .delete() \
                .where(StorageTableMetaModel.f_name == self.name,
                       StorageTableMetaModel.f_namespace == self.namespace) \
                .execute()


    @classmethod
    def create_address(cls, storage_engine, address_dict):
        return Relationship.EngineToAddress.get(storage_engine)(**address_dict)

    def get_name(self):
        return self.name

    def get_namespace(self):
        return self.namespace

    def get_address(self):
        return self.address

    def get_engine(self):
        return self.engine

    def get_type(self):
        return self.type

    def get_options(self):
        return self.options

    def get_partitions(self):
        return self.partitions

    def get_schema(self):
        return self.schema

    def get_count(self):
        return self.count

    def get_part_of_data(self):
        return self.part_of_data

    def get_description(self):
        return self.description
