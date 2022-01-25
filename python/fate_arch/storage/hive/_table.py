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
import uuid

from fate_arch.common import hive_utils
from fate_arch.common.file_utils import get_project_base_directory
from fate_arch.storage import StorageEngine, HiveStoreType
from fate_arch.storage import StorageTableBase


class StorageTable(StorageTableBase):
    def __init__(
        self,
        cur,
        con,
        address=None,
        name: str = None,
        namespace: str = None,
        partitions: int = 1,
        storage_type: HiveStoreType = HiveStoreType.DEFAULT,
        options=None,
    ):
        super(StorageTable, self).__init__(
            name=name,
            namespace=namespace,
            address=address,
            partitions=partitions,
            options=options,
            engine=StorageEngine.HIVE,
            store_type=storage_type,
        )
        self._cur = cur
        self._con = con

    def execute(self, sql, select=True):
        self._cur.execute(sql)
        if select:
            while True:
                result = self._cur.fetchone()
                if result:
                    yield result
                else:
                    break
        else:
            result = self._cur.fetchall()
            return result

    def _count(self, **kwargs):
        sql = 'select count(*) from {}'.format(self._address.name)
        try:
            self._cur.execute(sql)
            self._con.commit()
            ret = self._cur.fetchall()
            count = ret[0][0]
        except BaseException:
            count = 0
        return count

    def _collect(self, **kwargs) -> list:
        sql = "select * from {}".format(self._address.name)
        data = self.execute(sql)
        for line in data:
            yield hive_utils.deserialize_line(line)

    def _read(self) -> list:
        id_name, feature_name_list, _ = self._get_id_feature_name()
        id_feature_name = [id_name]
        id_feature_name.extend(feature_name_list)
        sql = "select {} from {}".format(",".join(id_feature_name), self._address.name)
        data = self.execute(sql)
        for line in data:
            yield hive_utils.read_line(line)

    def _put_all(self, kv_list, **kwargs):
        id_name, feature_name_list, id_delimiter = self.get_id_feature_name()
        create_table = "create table if not exists {}(k varchar(128) NOT NULL, v string) row format delimited fields terminated by" \
                       " '{}'".format(self._address.name, id_delimiter)
        self._cur.execute(create_table)
        # load local file or hdfs file
        temp_path = os.path.join(get_project_base_directory(), 'temp_data', uuid.uuid1().hex)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, 'w') as f:
            for k, v in kv_list:
                f.write(hive_utils.serialize_line(k, v))
        sql = "load data local inpath '{}' into table {}".format(temp_path, self._address.name)
        self._cur.execute(sql)
        self._con.commit()
        os.remove(temp_path)

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

    def _destroy(self):
        sql = "drop table {}".format(self._name)
        return self.execute(sql)

    def _save_as(self, address, name, namespace, partitions=None, **kwargs):
        sql = "create table {}.{} like {}.{};".format(namespace, name, self._namespace, self._name)
        return self.execute(sql)

    def check_address(self):
        schema = self.meta.get_schema()
        if schema:
            sql = 'SELECT {},{} FROM {}'.format(schema.get('sid'), schema.get('header'), self._address.name)
            feature_data = self.execute(sql)
            for feature in feature_data:
                if feature:
                    return True
        return False

    @staticmethod
    def get_meta_header(feature_name_list):
        create_features = ''
        feature_list = []
        feature_size = "varchar(255)"
        for feature_name in feature_name_list:
            create_features += '{} {},'.format(feature_name, feature_size)
            feature_list.append(feature_name)
        return create_features, feature_list

    def _get_id_feature_name(self):
        id = self.meta.get_schema().get("sid", "id")
        header = self.meta.get_schema().get("header")
        id_delimiter = self.meta.get_id_delimiter()
        if header:
            if isinstance(header, str):
                feature_list = header.split(id_delimiter)
            elif isinstance(header, list):
                feature_list = header
            else:
                feature_list = [header]
        else:
            raise Exception("mysql table need data header")
        return id, feature_list, id_delimiter
