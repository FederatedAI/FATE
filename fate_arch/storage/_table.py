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

import peewee
from fate_arch.common.log import getLogger
from fate_arch.common.base_utils import current_timestamp
from fate_arch.storage.metastore.db_models import DB, StorageTableMetaModel
from fate_arch.abc import StorageTableABC


MAX_NUM = 10000

LOGGER = getLogger()


class StorageTableBase(StorageTableABC):
    def __init__(self, name, namespace):
        self._name = name
        self._namespace = namespace
        self._meta = StorageTableMeta(name=self._name, namespace=self._namespace)

    def save_as(self, name, namespace, partitions=None, schema=None, **kwargs):
        if schema:
            self._meta.update_metas(name=name, namespace=namespace, schema=schema, partitions=partitions)

    def destroy(self):
        # destroy schema
        self._meta.destroy_metas()
        # subclass method needs do: super().destroy()

    def get_meta(self):
        return self._meta


class StorageTableMeta(object):
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
        self.init_meta()

    def init_meta(self):
        if not self.name or not self.namespace:
            raise RuntimeError("name and namespace cannot be empty")
        tables_meta = self.query_table_meta(filter_fields=dict(name=self.name, namespace=self.namespace))
        if not tables_meta:
            raise RuntimeError(f"can not found table meta by {self.name} and {self.namespace}")
        table_meta = tables_meta[0]
        for k, v in table_meta.__dict__["__data__"].items():
            setattr(self, k.lstrip("f_"), v)

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
                    setattr(table_meta, attr_name, v)
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
            for f_n in query_fields:
                attr_name = 'f_%s' % f_n
                if hasattr(StorageTableMetaModel, attr_name):
                    querys.append(operator.attrgetter('f_%s' % f_n)(StorageTableMetaModel))
            if filters:
                tables_meta = StorageTableMetaModel.select(querys).where(*filters)
                return [table_meta for table_meta in tables_meta]
            else:
                # not allow query all table
                return []

    """
    def get_meta(self, meta_type=StorageTableMetaType.SCHEMA):
        tables_meta = self.query_table_meta(filter_fields=dict(name=self.name, namespace=self.namespace), query_fields=[meta_type])
        if tables_meta:
            return getattr(f"f_{meta_type}", tables_meta[0])
        else:
            return None
    """

    @classmethod
    def update_metas(cls, name, namespace, schema=None, count=None, part_of_data=None, description=None, partitions=None, **kwargs):
        meta_info = {}
        for k, v in locals().items():
            if k not in ["self", "kwargs", "meta_info"] and v is not None:
                meta_info[k] = v
        meta_info.update(kwargs)
        meta_info["name"] = meta_info.get("name", name)
        meta_info["namespace"] = meta_info.get("namespace", namespace)
        with DB.connection_context():
            query_filters = []
            primary_keys = StorageTableMetaModel._meta.primary_key.field_names
            for p_k in primary_keys:
                query_filters.append(operator.attrgetter(p_k)(StorageTableMetaModel) == meta_info[p_k.lstrip("f_")])
            tables_meta = StorageTableMetaModel.select().where(*query_filters)
            if tables_meta:
                table_meta = tables_meta[0]
            else:
                raise Exception(f"can not found the {StorageTableMetaModel.__class__.__name__}")
            update_filters = query_filters[:]
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
        try:
            with DB.connection_context():
                StorageTableMetaModel \
                    .delete() \
                    .where(StorageTableMetaModel.f_name == self.name,
                           StorageTableMetaModel.f_namespace == self.namespace) \
                    .execute()
        except Exception as e:
            raise e
