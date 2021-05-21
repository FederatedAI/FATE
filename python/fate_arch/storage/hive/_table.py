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

from fate_arch.storage import StorageEngine, HiveStorageType
from fate_arch.storage import StorageTableBase


class StorageTable(StorageTableBase):
    def __init__(self,
                 cur,
                 con,
                 address=None,
                 name: str = None,
                 namespace: str = None,
                 partitions: int = 1,
                 storage_type: HiveStorageType = None,
                 options=None):
        super(StorageTable, self).__init__(name=name, namespace=namespace)
        self.cur = cur
        self.con = con
        self._address = address
        self._name = name
        self._namespace = namespace
        self._partitions = partitions
        self._options = options if options else {}
        self._storage_engine = StorageEngine.HIVE
        self._type = storage_type if storage_type else HiveStorageType.DEFAULT

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

    def count(self, **kwargs):
        sql = 'select count(*) from {}'.format(self._address.name)
        try:
            self.cur.execute(sql)
            self.con.commit()
            ret = self.cur.fetchall()
            count = ret[0][0]
        except:
            count = 0
        self.get_meta().update_metas(count=count)
        return count

    def collect(self, **kwargs) -> list:
        sql = 'select * from {}'.format(self._address.name)
        data = self.execute(sql)
        for i in data:
            yield i[0], self.get_meta().get_id_delimiter().join(list(i[1:]))

    def put_all(self, kv_list, **kwargs):
        pass

    def destroy(self):
        super().destroy()
        sql = 'drop table {}'.format(self._name)
        return self.execute(sql)

    @staticmethod
    def get_meta_header(feature_name_list, feature_num):
        create_features = ''
        feature_list = []
        if feature_name_list:
            for feature_name in feature_name_list:
                create_features += '{} LONGTEXT,'.format(feature_name)
                feature_list.append(feature_name)
        else:
            for i in range(0, feature_num):
                create_features += '{} LONGTEXT,'.format(f'feature_{i}')
                feature_list.append(f'feature_{i}')
        return create_features, feature_list
