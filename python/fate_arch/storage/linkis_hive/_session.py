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
from fate_arch.common.address import LinkisHiveAddress
from fate_arch.storage import StorageSessionBase, StorageEngine, LinkisHiveStoreType
from fate_arch.abc import AddressABC


class StorageSession(StorageSessionBase):
    def __init__(self, session_id, options=None):
        super(StorageSession, self).__init__(session_id=session_id, engine=StorageEngine.LINKIS_HIVE)
        self.con = None
        self.cur = None
        self.address = None

    def table(self, name, namespace, address: AddressABC, partitions,
              storage_type: LinkisHiveStoreType = LinkisHiveStoreType.DEFAULT, options=None, **kwargs):
        self.address = address
        if isinstance(address, LinkisHiveAddress):
            from fate_arch.storage.linkis_hive._table import StorageTable
            return StorageTable(
                address=address,
                name=name,
                namespace=namespace,
                storage_type=storage_type,
                partitions=partitions,
                options=options)
        raise NotImplementedError(f"address type {type(address)} not supported with eggroll storage")

    def cleanup(self, name, namespace):
        pass

    def stop(self):
        pass

    def kill(self):
        pass
