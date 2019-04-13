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
import os
from playhouse.pool import PooledMySQLDatabase
from peewee import Model
from arch.api.settings import DATABASES, USE_DATABASE
from arch.api.utils import log_utils

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
class DB(object):
    def __init__(self):
        data_base = DATABASES.get(USE_DATABASE)
        self.mysql_db = PooledMySQLDatabase(data_base.get('name'), **data_base)


def close_db(db):
    try:
        if db:
            db.close()
    except Exception as e:
        LOGGER.exception(e)


class BaseModel(Model):
    class Meta:
        database = None

    def to_json(self):
        return self.__dict__['__data__']
