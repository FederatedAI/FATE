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
import traceback

from impala.dbapi import connect

from fate_arch.common.address import HiveAddress
from fate_arch.storage import StorageSessionBase, StorageEngine, HiveStoreType
from fate_arch.abc import AddressABC


class StorageSession(StorageSessionBase):
    def __init__(self, session_id, options=None):
        super(StorageSession, self).__init__(session_id=session_id, engine=StorageEngine.HIVE)
        self._db_con = {}

    def table(self, name, namespace, address: AddressABC, partitions,
              storage_type: HiveStoreType = HiveStoreType.DEFAULT, options=None, **kwargs):

        if isinstance(address, HiveAddress):
            from fate_arch.storage.hive._table import StorageTable
            address_key = HiveAddress(
                host=address.host,
                username=None,
                port=address.port,
                database=address.database,
                auth_mechanism=None,
                password=None,
                name=None)
            if address_key in self._db_con:
                con, cur = self._db_con[address_key]
            else:
                self._create_db_if_not_exists(address)
                con = connect(host=address.host,
                              port=address.port,
                              database=address.database,
                              auth_mechanism=address.auth_mechanism,
                              password=address.password,
                              user=address.username
                              )
                cur = con.cursor()
                self._db_con[address_key] = (con, cur)
            return StorageTable(cur=cur, con=con, address=address, name=name, namespace=namespace,
                                storage_type=storage_type, partitions=partitions, options=options)
        raise NotImplementedError(f"address type {type(address)} not supported with eggroll storage")

    def cleanup(self, name, namespace):
        pass

    def stop(self):
        try:
            for key, val in self._db_con.items():
                con = val[0]
                cur = val[1]
                cur.close()
                con.close()
        except Exception as e:
            traceback.print_exc()

    def kill(self):
        return self.stop()

    def _create_db_if_not_exists(self, address):
        connection = connect(host=address.host,
                             port=address.port,
                             user=address.username,
                             auth_mechanism=address.auth_mechanism,
                             password=address.password
                             )
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("create database if not exists {}".format(address.database))
                print('create db {} success'.format(address.database))
            connection.commit()
