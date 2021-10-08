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

from fate_arch.storage import StorageEngine, HiveStoreType
from fate_arch.storage import StorageTableBase


class StorageTable(StorageTableBase):
    def __init__(self,
                 cur,
                 con,
                 address=None,
                 name: str = None,
                 namespace: str = None,
                 partitions: int = 1,
                 storage_type: HiveStoreType = None,
                 options=None):
        super(StorageTable, self).__init__(name=name, namespace=namespace)
        self.cur = cur
        self.con = con
        self._address = address
        self._name = name
        self._namespace = namespace
        self._partitions = partitions
        self._options = options if options else {}
        self._engine = StorageEngine.HIVE
        self._store_type = storage_type if storage_type else HiveStoreType.DEFAULT
        self._engine = StorageEngine.HIVE
        self._store_type = storage_type if storage_type else HiveStoreType.DEFAULT

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
    def engine(self):
        return self._engine

    @property
    def store_type(self):
        return self._store_type

    @property
    def partitions(self):
        return self._partitions

    @property
    def options(self):
        return self._options

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

    def count(self, **kwargs):
        sql = 'select count(*) from {}'.format(self._address.name)
        try:
            self.cur.execute(sql)
            self.con.commit()
            ret = self.cur.fetchall()
            count = ret[0][0]
        except:
            count = 0
        self.meta.update_metas(count=count)
        return count

    def collect(self, **kwargs) -> list:
        id_name, feature_name_list, _ = self.get_id_feature_name()
        id_feature_name = [id_name]
        id_feature_name.extend(feature_name_list)
        sql = 'select {} from {}'.format(','.join(id_feature_name), self._address.name)
        data = self.execute(sql)
        for line in data:
            feature_list = [str(feature) for feature in list(line[1:])]
            yield line[0], self.meta.get_id_delimiter().join(feature_list)

    def put_all(self, kv_list, **kwargs):
        id_name, feature_name_list, id_delimiter = self.get_id_feature_name()
        feature_sql, feature_list = StorageTable.get_meta_header(feature_name_list)
        id_size = "varchar(100)"
        create_table = "create table if not exists {}({} {} NOT NULL, {}) row format delimited fields terminated by" \
                       " ','".format(self._address.name, id_name, id_size, feature_sql.strip(','))
        self.cur.execute(create_table)
        sql = 'INSERT INTO {}({}, {})  VALUES'.format(self._address.name, id_name, ','.join(feature_list))
        for kv in kv_list:
            sql += '("{}", "{}"),'.format(kv[0], '", "'.join(kv[1].split(id_delimiter)))
        sql = ','.join(sql.split(',')[:-1])
        self.cur.execute(sql)
        self.con.commit()

    def get_id_feature_name(self):
        id = self.meta.get_schema().get('sid', 'id')
        header = self.meta.get_schema().get('header')
        id_delimiter = self.meta.get_id_delimiter()
        if header:
            if isinstance(header, str):
                feature_list = header.split(id_delimiter)
            elif isinstance(header, list):
                feature_list = header
            else:
                feature_list = [header]
        else:
            raise Exception("hive table need data header")
        return id, feature_list, id_delimiter

    def destroy(self):
        super().destroy()
        sql = 'drop table {}'.format(self._address.name)
        self.cur.execute(sql)
        self.con.commit()

    def check_address(self):
        schema = self.meta.get_schema()
        if schema:
            sql = 'SELECT {},{} FROM {}'.format(schema.get('sid'), schema.get('header'), self._address.name)
            feature_data = self.execute(sql)
            for feature in feature_data:
                if feature:
                    break
        return True

    @staticmethod
    def get_meta_header(feature_name_list):
        create_features = ''
        feature_list = []
        feature_size = "varchar(255)"
        for feature_name in feature_name_list:
            create_features += '{} {},'.format(feature_name, feature_size)
            feature_list.append(feature_name)
        return create_features, feature_list
