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

from fate_arch.federation import FederationEngine
from fate_arch.metastore.base_model import DateTimeField
from fate_arch.common import file_utils, log, EngineType, conf_utils
from fate_arch.common.conf_utils import decrypt_database_config
from fate_arch.metastore.base_model import JSONField, SerializedField, BaseModel


LOGGER = log.getLogger()

DATABASE = decrypt_database_config()
is_standalone = conf_utils.get_base_config("default_engines", {}).get(EngineType.FEDERATION).upper() == \
    FederationEngine.STANDALONE


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
        if is_standalone:
            from playhouse.apsw_ext import APSWDatabase
            self.database_connection = APSWDatabase(file_utils.get_project_base_directory("fate_sqlite.db"))
        else:
            from playhouse.pool import PooledMySQLDatabase
            self.database_connection = PooledMySQLDatabase(db_name, **database_config)


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
    create_failed_list = []
    for name, obj in members:
        if obj != DataBaseModel and issubclass(obj, DataBaseModel):
            table_objs.append(obj)
            LOGGER.info(f"start create table {obj.__name__}")
            try:
                obj.create_table()
                LOGGER.info(f"create table success: {obj.__name__}")
            except Exception as e:
                LOGGER.exception(e)
                create_failed_list.append(obj.__name__)
    if create_failed_list:
        LOGGER.info(f"create tables failed: {create_failed_list}")
        raise Exception(f"create tables failed: {create_failed_list}")


class StorageConnectorModel(DataBaseModel):
    f_name = CharField(max_length=100, primary_key=True)
    f_engine = CharField(max_length=100, index=True)  # 'MYSQL'
    f_connector_info = JSONField()

    class Meta:
        db_table = "t_storage_connector"


class StorageTableMetaModel(DataBaseModel):
    f_name = CharField(max_length=100, index=True)
    f_namespace = CharField(max_length=100, index=True)
    f_address = JSONField()
    f_engine = CharField(max_length=100)  # 'EGGROLL', 'MYSQL'
    f_store_type = CharField(max_length=50, null=True)  # store type
    f_options = JSONField()
    f_partitions = IntegerField(null=True)

    f_id_delimiter = CharField(null=True)
    f_in_serialized = BooleanField(default=True)
    f_have_head = BooleanField(default=True)
    f_extend_sid = BooleanField(default=False)
    f_auto_increasing_sid = BooleanField(default=False)

    f_schema = SerializedField()
    f_count = BigIntegerField(null=True)
    f_part_of_data = SerializedField()
    f_origin = CharField(max_length=50, default='')
    f_disable = BooleanField(default=False)
    f_description = TextField(default='')

    f_read_access_time = BigIntegerField(null=True)
    f_read_access_date = DateTimeField(null=True)
    f_write_access_time = BigIntegerField(null=True)
    f_write_access_date = DateTimeField(null=True)

    class Meta:
        db_table = "t_storage_table_meta"
        primary_key = CompositeKey('f_name', 'f_namespace')


class SessionRecord(DataBaseModel):
    f_engine_session_id = CharField(max_length=150, null=False)
    f_manager_session_id = CharField(max_length=150, null=False)
    f_engine_type = CharField(max_length=10, index=True)
    f_engine_name = CharField(max_length=50, index=True)
    f_engine_address = JSONField()

    class Meta:
        db_table = "t_session_record"
        primary_key = CompositeKey("f_engine_type", "f_engine_name", "f_engine_session_id")
