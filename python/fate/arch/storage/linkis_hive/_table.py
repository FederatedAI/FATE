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

from fate_arch.storage import StorageEngine, LinkisHiveStoreType
from fate_arch.storage import StorageTableBase
from fate_arch.storage.linkis_hive._settings import (
    Token_Code,
    Token_User,
    STATUS_URI,
    EXECUTE_URI,
)


class StorageTable(StorageTableBase):
    def __init__(
        self,
        address=None,
        name: str = None,
        namespace: str = None,
        partitions: int = 1,
        storage_type: LinkisHiveStoreType = LinkisHiveStoreType.DEFAULT,
        options=None,
    ):
        super(StorageTable, self).__init__(
            name=name,
            namespace=namespace,
            address=address,
            partitions=partitions,
            options=options,
            engine=StorageEngine.LINKIS_HIVE,
            store_type=storage_type,
        )

    def _count(self, **kwargs):
        sql = "select count(*) from {}".format(self._address.name)
        try:
            count = self.execute(sql)
        except BaseException:
            count = 0
        return count

    def _collect(self, **kwargs):
        if kwargs.get("is_spark"):
            from pyspark.sql import SparkSession

            session = SparkSession.builder.enableHiveSupport().getOrCreate()
            data = session.sql(
                f"select * from {self._address.database}.{self._address.name}"
            )
            return data
        else:
            sql = "select * from {}.{}".format(
                self._address.database, self._address.name
            )
            data = self.execute(sql)
            for i in data:
                yield i[0], self.meta.get_id_delimiter().join(list(i[1:]))

    def _put_all(self, kv_pd, **kwargs):
        from pyspark.sql import SparkSession

        session = SparkSession.builder.enableHiveSupport().getOrCreate()
        session.sql("use {}".format(self._address.database))
        spark_df = session.createDataFrame(kv_pd)
        spark_df.write.saveAsTable(self._address.name, format="orc")

    def _destroy(self):
        sql = "drop table {}.{}".format(self._address.database, self._address.name)
        return self.execute(sql)

    def _save_as(self, address, name, namespace, partitions, **kwargs):
        pass

    def execute(self, sql):
        exec_id = self._execute_entrance(sql)
        while True:
            status = self._status_entrance(exec_id)
            if status:
                break
            time.sleep(1)
        return self._result_entrance()

    def _execute_entrance(self, sql):
        execute_url = f"http://{self._address.host}:{self._address.port}{EXECUTE_URI}"
        data = {
            "method": EXECUTE_URI,
            "params": self._address.params,
            "executeApplicationName": self._address.execute_application_name,
            "executionCode": sql,
            "runType": self._address.run_type,
            "source": self._address.source,
        }
        # token
        headers = {
            "Token-Code": Token_Code,
            "Token-User": Token_User,
            "Content-Type": "application/json",
        }
        execute_response = requests.post(url=execute_url, headers=headers, json=data)
        if execute_response.json().get("status") == 0:
            return execute_response.json()["data"]["execID"]
        else:
            raise SystemError(
                f"request linkis hive execue entrance failed, status: {execute_response.json().get('status')},"
                f" message: {execute_response.json().get('message')}"
            )

    def _status_entrance(self, exec_id):
        execute_url = (
            f"http://{self._address.host}:{self._address.port}{STATUS_URI}".replace(
                "exec_id", exec_id
            )
        )
        headers = {
            "Token-Code": "MLSS",
            "Token-User": "alexwu",
            "Content-Type": "application/json",
        }
        execute_response = requests.Session().get(url=execute_url, headers=headers)
        if execute_response.json().get("status") == 0:
            execute_status = execute_response.json()["data"]["status"]
            if execute_status == "Success":
                return True
            elif execute_status == "Failed":
                raise Exception(
                    f"request linkis hive status entrance failed, status: {execute_status}"
                )
            else:
                return False
        else:
            raise SystemError(
                f"request linkis hive status entrance failed, status: {execute_response.json().get('status')},"
                f" message: {execute_response.json().get('message')}"
            )

    def _result_entrance(self):
        pass
