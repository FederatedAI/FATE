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
import inspect
import os
import sys

from peewee import CharField, IntegerField, BigIntegerField, TextField, CompositeKey, BooleanField
from playhouse.apsw_ext import APSWDatabase
from playhouse.pool import PooledMySQLDatabase

from fate_arch.storage.metastore.base_model import JSONField, SerializedField, BaseModel
from fate_arch.common.conf_utils import get_base_config
from fate_arch.common import WorkMode, file_utils
from fate_arch.common import log

DATABASE = get_base_config("database", {})
WORK_MODE = get_base_config('work_mode', 0)

LOGGER = log.getLogger()


def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        key = str(cls) + str(os.getpid())
        if key not in instances:
            instances[key] = cls(*args, **kw)
        return instances[key]

    return _singleton


@singleton
class BaseDataBase(object):
    def __init__(self):
        database_config = DATABASE.copy()
        db_name = database_config.pop("name")
        if WORK_MODE == WorkMode.STANDALONE:
            db_file_path = os.path.join(file_utils.get_python_base_directory(), 'fate_flow', 'fate_flow_sqlite.db')
            self.database_connection = APSWDatabase(db_file_path)
        elif WORK_MODE == WorkMode.CLUSTER:
            self.database_connection = PooledMySQLDatabase(db_name, **database_config)
        else:
            raise Exception('can not init database')


DB = BaseDataBase().database_connection


def close_connection():
    try:
        if DB:
            DB.close()
    except Exception as e:
        LOGGER.exception(e)


class DataBaseModel(BaseModel):
    class Meta:
        database = DB


@DB.connection_context()
def init_database_tables():
    members = inspect.getmembers(sys.modules[__name__], inspect.isclass)
    table_objs = []
    for name, obj in members:
        if obj != DataBaseModel and issubclass(obj, DataBaseModel):
            table_objs.append(obj)
    DB.create_tables(table_objs)


class StorageTableMetaModel(DataBaseModel):
    f_name = CharField(max_length=100, index=True)
    f_namespace = CharField(max_length=100, index=True)
    f_address = JSONField()
    f_engine = CharField(max_length=100, index=True)  # 'EGGROLL', 'MYSQL'
    f_type = CharField(max_length=50, index=True)  # storage type
    f_options = JSONField()
    f_partitions = IntegerField(null=True)

    f_id_delimiter = CharField(null=True)
    f_in_serialized = BooleanField(default=True)
    f_have_head = BooleanField(default=True)

    f_schema = SerializedField()
    f_count = IntegerField(null=True)
    f_part_of_data = SerializedField()
    f_description = TextField(default='')

    f_create_time = BigIntegerField()
    f_update_time = BigIntegerField(null=True)

    class Meta:
        db_table = "t_storage_table_meta"
        primary_key = CompositeKey('f_name', 'f_namespace')


class SessionRecord(DataBaseModel):
    f_session_id = CharField(max_length=150, null=False, primary_key=True)
    f_engine_name = CharField(max_length=50, index=True)
    f_engine_type = CharField(max_length=10, index=True)
    f_engine_address = JSONField()
    f_create_time = BigIntegerField(index=True)

    class Meta:
        db_table = "t_session_record"
