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
from impala.dbapi import connect

from fate_arch.common.address import HiveAddress
from fate_arch.storage import StorageSessionBase, StorageEngine, HiveStoreType
from fate_arch.abc import AddressABC


class StorageSession(StorageSessionBase):
    def __init__(self, session_id, options=None):
        super(StorageSession, self).__init__(session_id=session_id, engine_name=StorageEngine.HIVE)
        self.con = None
        self.cur = None
        self.address = None

    def create(self):
        pass

    def table(self, name, namespace, address: AddressABC, partitions,
              storage_type: HiveStoreType = HiveStoreType.DEFAULT, options=None, **kwargs):
        self.address = address
        if isinstance(address, HiveAddress):
            from fate_arch.storage.hive._table import StorageTable
            self.create_db()
            self.con = connect(host=address.host,
                               port=address.port,
                               database=address.database,
                               auth_mechanism=address.auth_mechanism,
                               password=address.password,
                               user=address.username
                               )
            self.cur = self.con.cursor()
            return StorageTable(cur=self.cur, con=self.con, address=address, name=name, namespace=namespace,
                                storage_type=storage_type, partitions=partitions, options=options)
        raise NotImplementedError(f"address type {type(address)} not supported with eggroll storage")

    def cleanup(self, name, namespace):
        pass

    def stop(self):
        return self.con.close()

    def kill(self):
        return self.con.close()

    def create_db(self):
        conn = connect(host=self.address.host,
                       port=self.address.port,
                       user=self.address.username,
                       auth_mechanism=self.address.auth_mechanism,
                       password=self.address.password
                       )
        cursor = conn.cursor()
        cursor.execute("create database if not exists {}".format(self.address.database))
        print('create db {} success'.format(self.address.database))
        conn.close()
