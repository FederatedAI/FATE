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
import time

import requests

from fate_arch.storage import StorageEngine, LinkisHiveStorageType
from fate_arch.storage import StorageTableBase
from fate_arch.storage.linkis_hive._settings import Token_Code, Token_User, STATUS_URI, EXECUTE_URI


class StorageTable(StorageTableBase):
    def __init__(self,
                 address=None,
                 name: str = None,
                 namespace: str = None,
                 partitions: int = 1,
                 storage_type: LinkisHiveStorageType = None,
                 options=None):
        super(StorageTable, self).__init__(name=name, namespace=namespace)
        self._address = address
        self._name = name
        self._namespace = namespace
        self._partitions = partitions
        self._options = options if options else {}
        self._storage_engine = StorageEngine.LINKIS_HIVE
        self._type = storage_type if storage_type else LinkisHiveStorageType.DEFAULT

    def execute_entrance(self, sql):
        execute_url = f"http://{self._address.host}:{self._address.port}{EXECUTE_URI}"
        data = {
            "method": EXECUTE_URI,
            "params": self._address.params,
            "executeApplicationName": self._address.execute_application_name,
            "executionCode": sql,
            "runType": self._address.run_type,
            "source": self._address.source
        }
        # token
        headers = {"Token-Code": Token_Code, "Token-User": Token_User, "Content-Type": "application/json"}
        execute_response = requests.post(url=execute_url, headers=headers, json=data)
        if execute_response.json().get("status") == 0:
            return execute_response.json()["data"]["execID"]
        else:
            raise SystemError(f"request linkis hive execue entrance failed, status: {execute_response.json().get('status')},"
                              f" message: {execute_response.json().get('message')}")

    def status_entrance(self, exec_id):
        execute_url = f"http://{self._address.host}:{self._address.port}{STATUS_URI}".replace("exec_id", exec_id)
        headers = {"Token-Code": "MLSS", "Token-User": "alexwu", "Content-Type": "application/json"}
        execute_response = requests.Session().get(url=execute_url, headers=headers)
        if execute_response.json().get("status") == 0:
            execute_status = execute_response.json()["data"]["status"]
            if execute_status == "Success":
                return True
            elif execute_status == "Failed":
                raise Exception(f"request linkis hive status entrance failed, status: {execute_status}")
            else:
                return False
        else:
            raise SystemError(f"request linkis hive status entrance failed, status: {execute_response.json().get('status')},"
                              f" message: {execute_response.json().get('message')}")

    def result_entrance(self):
        pass

    def execute(self, sql):
        exec_id = self.execute_entrance(sql)
        while True:
            status = self.status_entrance(exec_id)
            if status:
                break
            time.sleep(1)
        return self.result_entrance()

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
            count = self.execute(sql)
        except:
            count = 0
        self.get_meta().update_metas(count=count)
        return count

    def collect(self, **kwargs):
        from fate_arch.common.log import schedule_logger
        if kwargs.get("is_spark"):
            from pyspark.sql import SparkSession
            session = SparkSession.builder.enableHiveSupport().getOrCreate()
            data = session.sql(f"select * from {self._address.database}.{self._address.name}")
            schedule_logger('wzh_test').info(data)
            return data
        else:
            schedule_logger('wzh_test').info(f"no spark")
            sql = 'select * from {}.{}'.format(self._address.database, self._address.name)
            data = self.execute(sql)
            for i in data:
                yield i[0], self.get_meta().get_id_delimiter().join(list(i[1:]))

    def put_all(self, kv_pd, **kwargs):
        from pyspark.sql import SparkSession
        session = SparkSession.builder.enableHiveSupport().getOrCreate()
        session.sql("use {}".format(self._address.database))
        spark_df = session.createDataFrame(kv_pd)
        spark_df.write.saveAsTable(self._address.name, format="orc")

    def destroy(self):
        super().destroy()
        sql = 'drop table {}.{}'.format(self._address.database, self._address.name)
        return self.execute(sql)