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
from fate_arch.relation_ship import Relationship
from fate_arch.metastore.db_models import DB, StorageTableMetaModel

LOGGER = getLogger()


class StorageTableBase(StorageTableABC):
    def __init__(self, name, namespace, address, partitions, options, engine, store_type):
        self._name = name
        self._namespace = namespace
        self._address = address
        self._partitions = partitions
        self._options = options if options else {}
        self._engine = engine
        self._store_type = store_type

        self._meta = None
        self._read_access_time = None
        self._write_access_time = None

    @property
    def name(self):
        return self._name

    @property
    def namespace(self):
        return self._namespace

    @property
    def address(self):
        return self._address

    @property
    def partitions(self):
        return self._partitions

    @property
    def options(self):
        return self._options

    @property
    def engine(self):
        return self._engine

    @property
    def store_type(self):
        return self._store_type

    @property
    def meta(self):
        return self._meta

    @meta.setter
    def meta(self, meta):
        self._meta = meta

    @property
    def read_access_time(self):
        return self._read_access_time

    @property
    def write_access_time(self):
        return self._write_access_time

    def update_meta(self,
                    schema=None,
                    count=None,
                    part_of_data=None,
                    description=None,
                    partitions=None,
                    **kwargs):
        self._meta.update_metas(schema=schema,
                                count=count,
                                part_of_data=part_of_data,
                                description=description,
                                partitions=partitions,
                                **kwargs)

    def create_meta(self, **kwargs):
        table_meta = StorageTableMeta(name=self._name, namespace=self._namespace, new=True)
        table_meta.set_metas(**kwargs)
        table_meta.address = self._address
        table_meta.partitions = self._partitions
        table_meta.engine = self._engine
        table_meta.store_type = self._store_type
        table_meta.options = self._options
        table_meta.create()
        self._meta = table_meta

        return table_meta

    def check_address(self):
        return True

    def put_all(self, kv_list: Iterable, **kwargs):
        self._update_write_access_time()
        self._put_all(kv_list, **kwargs)

    def collect(self, **kwargs) -> list:
        self._update_read_access_time()
        return self._collect(**kwargs)

    def count(self):
        self._update_read_access_time()
        count = self._count()
        self.meta.update_metas(count=count)
        return count

    def read(self):
        self._update_read_access_time()
        return self._read()

    def destroy(self):
        self.meta.destroy_metas()
        self._destroy()

    def save_as(self, address, name, namespace, partitions=None, **kwargs):
        table = self._save_as(address, name, namespace, partitions, **kwargs)
        table.create_meta(**kwargs)
        return table

    def _update_read_access_time(self, read_access_time=None):
        read_access_time = current_timestamp() if not read_access_time else read_access_time
        self._meta.update_metas(read_access_time=read_access_time)

    def _update_write_access_time(self, write_access_time=None):
        write_access_time = current_timestamp() if not write_access_time else write_access_time
        self._meta.update_metas(write_access_time=write_access_time)

    # to be implemented
    def _put_all(self, kv_list: Iterable, **kwargs):
        raise NotImplementedError()

    def _collect(self, **kwargs) -> list:
        raise NotImplementedError()

    def _count(self):
        raise NotImplementedError()

    def _read(self):
        raise NotImplementedError()

    def _destroy(self):
        raise NotImplementedError()

    def _save_as(self, address, name, namespace, partitions=None, schema=None, **kwargs):
        raise NotImplementedError()


class StorageTableMeta(StorageTableMetaABC):

    def __init__(self, name, namespace, new=False, create_address=True):
        self.name = name
        self.namespace = namespace
        self.address = None
        self.engine = None
        self.store_type = None
        self.options = None
        self.partitions = None
        self.in_serialized = None
        self.have_head = None
        self.id_delimiter = None
        self.extend_sid = False
        self.auto_increasing_sid = None
        self.schema = None
        self.count = None
        self.part_of_data = None
        self.description = None
        self.origin = None
        self.disable = None
        self.create_time = None
        self.update_time = None
        self.read_access_time = None
        self.write_access_time = None
        if self.options is None:
            self.options = {}
        if self.schema is None:
            self.schema = {}
        if self.part_of_data is None:
            self.part_of_data = []
        if not new:
            self.build(create_address)

    def build(self, create_address):
        for k, v in self.table_meta.__dict__["__data__"].items():
            setattr(self, k.lstrip("f_"), v)
        if create_address:
            self.address = self.create_address(storage_engine=self.engine, address_dict=self.address)

    def __new__(cls, *args, **kwargs):
        if not kwargs.get("new", False):
            name, namespace = kwargs.get("name"), kwargs.get("namespace")
            if not name or not namespace:
                return None
            tables_meta = cls.query_table_meta(filter_fields=dict(name=name, namespace=namespace))
            if not tables_meta:
                return None
            self = super().__new__(cls)
            setattr(self, "table_meta", tables_meta[0])
            return self
        else:
            return super().__new__(cls)

    def exists(self):
        if hasattr(self, "table_meta"):
            return True
        else:
            return False

    @DB.connection_context()
    def create(self):
        table_meta = StorageTableMetaModel()
        table_meta.f_create_time = current_timestamp()
        table_meta.f_schema = {}
        table_meta.f_part_of_data = []
        for k, v in self.to_dict().items():
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
            elif isinstance(e.args[0], str) and "UNIQUE constraint failed" in e.args[0]:
                pass
            else:
                raise e
        except Exception as e:
            raise e

    def set_metas(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

    @classmethod
    @DB.connection_context()
    def query_table_meta(cls, filter_fields, query_fields=None):
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

    @DB.connection_context()
    def update_metas(self, schema=None, count=None, part_of_data=None, description=None, partitions=None,
                     in_serialized=None, **kwargs):
        meta_info = {}
        for k, v in locals().items():
            if k not in ["self", "kwargs", "meta_info"] and v is not None:
                meta_info[k] = v
        meta_info.update(kwargs)
        meta_info["name"] = meta_info.get("name", self.name)
        meta_info["namespace"] = meta_info.get("namespace", self.namespace)
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
                        tmp = v
                    else:
                        tmp = v[:100]
                    update_fields[operator.attrgetter(attr_name)(StorageTableMetaModel)] = tmp
                else:
                    update_fields[operator.attrgetter(attr_name)(StorageTableMetaModel)] = v
        if update_filters:
            operate = table_meta.update(update_fields).where(*update_filters)
        else:
            operate = table_meta.update(update_fields)
        if count:
            self.count = count
        _return = operate.execute()
        _meta = StorageTableMeta(name=self.name, namespace=self.namespace)
        return _return > 0, _meta

    @DB.connection_context()
    def destroy_metas(self):
        StorageTableMetaModel \
            .delete() \
            .where(StorageTableMetaModel.f_name == self.name,
                   StorageTableMetaModel.f_namespace == self.namespace) \
            .execute()

    @classmethod
    def create_address(cls, storage_engine, address_dict):
        address_class = Relationship.EngineToAddress.get(storage_engine)
        kwargs = {}
        for k in address_class.__init__.__code__.co_varnames:
            if k == "self":
                continue
            if address_dict.get(k, None):
                kwargs[k] = address_dict[k]
        return address_class(**kwargs)

    def get_name(self):
        return self.name

    def get_namespace(self):
        return self.namespace

    def get_address(self):
        return self.address

    def get_engine(self):
        return self.engine

    def get_store_type(self):
        return self.store_type

    def get_options(self):
        return self.options

    def get_partitions(self):
        return self.partitions

    def get_in_serialized(self):
        return self.in_serialized

    def get_id_delimiter(self):
        return self.id_delimiter

    def get_extend_sid(self):
        return self.extend_sid

    def get_auto_increasing_sid(self):
        return self.auto_increasing_sid

    def get_have_head(self):
        return self.have_head

    def get_origin(self):
        return self.origin

    def get_disable(self):
        return self.disable

    def get_schema(self):
        return self.schema

    def get_count(self):
        return self.count

    def get_part_of_data(self):
        return self.part_of_data

    def get_description(self):
        return self.description

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if v is None or k == "table_meta":
                continue
            d[k] = v
        return d
