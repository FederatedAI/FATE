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

import uuid
import pymysql
import typing

from arch.api.base.utils.store_type import StoreTypes
from arch.api.utils.profile_util import log_elapsed
from arch.api import WorkMode

# noinspection SpellCheckingInspection,PyProtectedMember,PyPep8Naming
from fate_flow.manager.data_manager.data_table import Table


class MysqlTable(Table):
    def __init__(self,
                 mode: typing.Union[int, WorkMode] = WorkMode.STANDALONE,
                 persistent_engine: str = StoreTypes.MYSQL,
                 namespace: str = None,
                 name: str = None,
                 partition: int = 1,
                 database_config: dict = None,
                 **kwargs):
        self._name = name or str(uuid.uuid1())
        self._namespace = namespace or str(uuid.uuid1())
        self._partitions = partition
        self._strage_engine = persistent_engine
        self.database_config = database_config
        self._mode = mode
        '''
        database_config
        {
            'user': 'root',
            'passwd': 'fate_dev',
            'host': '127.0.0.1',
            'port': 3306,
            'charset': 'utf8'
        }
        '''
        try:
            self.con = pymysql.connect(host=database_config.get('host'),
                                       user=database_config.get('user'),
                                       passwd=database_config.get('passwd'),
                                       port=database_config.get('port'),
                                       db=namespace)
            self.cur = self.con.cursor()
        except:
            print("DataBase connect error,please check the db config.")

    def execute(self, sql, select=True):
        self.cur.execute(sql)
        if select:
            while True:
                result = self.cur.fetchone()
                if result:
                    yield result
                else:
                    break
        else:
            result = self.cur.fetchall()
            return result

    def get_name(self):
        return self._name

    def get_namespace(self):
        return self._namespace

    def get_partitions(self):
        return self._partitions

    def get_storage_engine(self):
        return self._strage_engine

    def get_address(self):
        return self.database_config

    @log_elapsed
    def collect(self, min_chunk_size=0, use_serialize=True, **kwargs) -> list:
        sql = 'select * from {}'.format(self._name)
        data = self.execute(sql)
        return data

    def destroy(self):
        sql = 'drop table {}'.format(self._name)
        return self.execute(sql)

    @log_elapsed
    def count(self, **kwargs):
        sql = 'select count(*) from {}'.format(self._name)
        return self.execute(sql)

    def quit(self):
        self.con.close()
