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

from fate_arch.common.profile import log_elapsed
from fate_arch.storage import StorageEngine, MySQLStorageType
from fate_arch.storage import StorageTableBase


class StorageTable(StorageTableBase):
    def __init__(self,
                 cur,
                 con,
                 address=None,
                 name: str = None,
                 namespace: str = None,
                 partitions: int = 1,
                 storage_type: MySQLStorageType = None,
                 options=None):
        self.cur = cur
        self.con = con
        self._address = address
        self._name = name
        self._namespace = namespace
        self._partitions = partitions
        self._storage_type = storage_type
        self._options = options if options else {}
        self._storage_engine = StorageEngine.MYSQL
        self._type = storage_type if storage_type else MySQLStorageType.InnoDB

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

    def get_partitions(self):
        return self._partitions

    def get_name(self):
        return self._name

    def get_namespace(self):
        return self._namespace

    def get_engine(self):
        return self._storage_engine

    def get_address(self):
        return self._address

    def get_type(self):
        return self._type

    def get_options(self):
        return self._options

    @log_elapsed
    def count(self, **kwargs):
        sql = 'select count(*) from {}'.format(self._name)
        try:
            self.cur.execute(sql)
            self.con.commit()
            ret = self.cur.fetchall()
            count = ret[0][0]
        except:
            count = 0
        return count

    @log_elapsed
    def collect(self, **kwargs) -> list:
        sql = 'select * from {}'.format(self._name)
        data = self.execute(sql)
        return data

    def put_all(self, kv_list, **kwargs):
        create_table = 'create table if not exists {}(id varchar(50) NOT NULL, features LONGTEXT, PRIMARY KEY(id))'.format(self._address.name)
        self.cur.execute(create_table)
        sql = 'REPLACE INTO {}(id, features)  VALUES'.format(self._address.name)
        for kv in kv_list:
            sql += '("{}", "{}"),'.format(kv[0], kv[1])
        sql = ','.join(sql.split(',')[:-1]) + ';'
        self.cur.execute(sql)
        self.con.commit()

    def destroy(self):
        super().destroy()
        sql = 'drop table {}'.format(self._name)
        return self.execute(sql)
