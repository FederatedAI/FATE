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

import __main__
from peewee import Model, CharField, IntegerField, BigIntegerField, TextField, CompositeKey
from playhouse.pool import PooledMySQLDatabase
from playhouse.apsw_ext import APSWDatabase

from arch.api.utils import log_utils
from arch.api.utils.core import current_timestamp
from fate_flow.entity.constant_config import WorkMode
from fate_flow.settings import DATABASE, USE_LOCAL_DATABASE, WORK_MODE, stat_logger
from fate_flow.entity.runtime_config import RuntimeConfig

LOGGER = log_utils.getLogger()


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
            if USE_LOCAL_DATABASE:
                self.database_connection = APSWDatabase('fate_flow_sqlite.db')
                RuntimeConfig.init_config(USE_LOCAL_DATABASE=True)
                stat_logger.info('init sqlite database on standalone mode successfully')
            else:
                self.database_connection = PooledMySQLDatabase(db_name, **database_config)
                stat_logger.info('init mysql database on standalone mode successfully')
                RuntimeConfig.init_config(USE_LOCAL_DATABASE=False)
        elif WORK_MODE == WorkMode.CLUSTER:
            self.database_connection = PooledMySQLDatabase(db_name, **database_config)
            stat_logger.info('init mysql database on cluster mode successfully')
            RuntimeConfig.init_config(USE_LOCAL_DATABASE=False)
        else:
            raise Exception('can not init database')

DB = BaseDataBase().database_connection

def close_connection(db_connection):
    try:
        if db_connection:
            db_connection.close()
    except Exception as e:
        LOGGER.exception(e)


class DataBaseModel(Model):
    class Meta:
        database = DB

    def to_json(self):
        return self.__dict__['__data__']

    def save(self, *args, **kwargs):
        if hasattr(self, "f_update_time"):
            self.f_update_time = current_timestamp()
        super(DataBaseModel, self).save(*args, **kwargs)


def init_database_tables():
    with DB.connection_context():
        members = inspect.getmembers(sys.modules[__name__], inspect.isclass)
        table_objs = []
        for name, obj in members:
            if obj != DataBaseModel and issubclass(obj, DataBaseModel):
                table_objs.append(obj)
        DB.create_tables(table_objs)



DEFAULT_TAG = 'Za'

class IdLibraryCacheInfo(DataBaseModel):
    f_party_id = CharField(max_length=32)
    f_id_type = CharField(max_length=16)
    f_encrypt_type = CharField(max_length=16)
    f_tag = CharField(max_length=16, default=DEFAULT_TAG)
    f_namespcae = CharField(max_length=128)
    f_version = CharField(max_length=128)
    f_rsa_key_n = CharField(max_length=512)
    f_rsa_key_d = CharField(max_length=512)
    f_rsa_key_e = CharField(max_length=32)
    f_create_time = BigIntegerField()
    f_update_time = BigIntegerField(null=True)
    f_description = TextField(null=True, default='')

    class Meta:
        db_table = "t_id_library_cache_infos"
        primary_key = CompositeKey('f_party_id', 'f_id_type', 'f_encrypt_type', 'f_tag', 'f_namespcae', 'f_version')
