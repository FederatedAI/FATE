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
import operator
import peewee
from fate_arch.common.log import getLogger
from fate_arch.storage._types import StorageTableMetaType
from fate_arch.common.core_utils import current_timestamp
from fate_arch.metastore.db_models import DB, StorageTableMeta
from fate_arch.abc import StorageTableABC

MAX_NUM = 10000

LOGGER = getLogger()


class StorageTableBase(StorageTableABC):
    def save_as(self, name, namespace, partition=None, schema=None, **kwargs):
        if schema:
            self.update_metas(name=name, namespace=namespace, schema=schema, partitions=partition)

    def destroy(self):
        # destroy schema
        self.destroy_metas()
        # subclass method needs do: super().destroy()

    def create_metas(self, **kwargs):
        with DB.connection_context():
            table_meta = StorageTableMeta()
            table_meta.f_create_time = current_timestamp()
            for k, v in kwargs.items():
                attr_name = 'f_%s' % k
                if hasattr(StorageTableMeta, attr_name):
                    if k == "part_of_data":
                        setattr(table_meta, attr_name, v[:100])
                    else:
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
                    raise Exception(f"create table meta failed:\n{e}")
            except Exception as e:
                raise Exception(f"create table meta failed:\n{e}")

    def get_metas(self, filter_fields, query_fields=None):
        with DB.connection_context():
            filters = []
            querys = []
            for f_n, f_v in filter_fields.items():
                attr_name = 'f_%s' % f_n
                if hasattr(StorageTableMeta, attr_name):
                    filters.append(operator.attrgetter('f_%s' % f_n)(StorageTableMeta) == f_v)
            for f_n in query_fields:
                attr_name = 'f_%s' % f_n
                if hasattr(StorageTableMeta, attr_name):
                    querys.append(operator.attrgetter('f_%s' % f_n)(StorageTableMeta))
            if filters:
                table_metas = StorageTableMeta.select(querys).where(*filters)
                return [table_meta for table_meta in table_metas]
            else:
                # not allow query all table
                return []

    def get_meta(self, meta_type=StorageTableMetaType.SCHEMA, name=None, namespace=None):
        if not name and not namespace:
            name = self.get_name()
            namespace = self.get_namespace()
        metas = self.get_metas(filter_fields=dict(name=name, namespace=namespace), query_fields=[meta_type])
        if metas:
            return getattr(f"f_{meta_type}", metas[0])
        else:
            return None

    def update_metas(self, **kwargs):
        kwargs["name"] = kwargs.get("name", self.get_name())
        kwargs["namespace"] = kwargs.get("namespace", self.get_namespace())
        with DB.connection_context():
            query_filters = []
            primary_keys = StorageTableMeta._meta.primary_key.field_names
            for p_k in primary_keys:
                query_filters.append(operator.attrgetter(p_k)(StorageTableMeta) == kwargs[p_k.lstrip("f_")])
            tables_meta = StorageTableMeta.select().where(*query_filters)
            if tables_meta:
                table_meta = tables_meta[0]
            else:
                raise Exception(f"can not found the {StorageTableMeta.__class__.__name__}")
            update_filters = query_filters[:]
            update_fields = {}
            for k, v in kwargs.items():
                attr_name = 'f_%s' % k
                if hasattr(kwargs, attr_name) and attr_name not in primary_keys:
                    if k == "part_of_data":
                        if len(v) < 100:
                            tmp = table_meta.f_part_of_data[- (100 - len(v)):] + v
                        else:
                            tmp = v[:100]
                        update_fields[operator.attrgetter(attr_name)(StorageTableMeta)] = tmp
                    else:
                        update_fields[operator.attrgetter(attr_name)(StorageTableMeta)] = v
            if update_filters:
                operate = table_meta.update(update_fields).where(*update_filters)
            else:
                operate = table_meta.update(update_fields)
            return operate.execute() > 0

    def destroy_metas(self):
        try:
            with DB.connection_context():
                StorageTableMeta \
                    .delete() \
                    .where(StorageTableMeta.f_name == self.get_name(),
                           StorageTableMeta.f_namespace == self.get_namespace()) \
                    .execute()
        except Exception as e:
            LOGGER.error("delete_table_meta {}, {}, exception:{}.".format(self.get_namespace(), self.get_name(), e))

    @abc.abstractmethod
    def get_address(self):
        pass
