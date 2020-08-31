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

from fate_arch.abc import AddressABC
from fate_arch.common.address import EggRollAddress
from fate_arch.common.profile import log_elapsed
from fate_arch.storage import StorageSessionBase, EggRollStorageType


class StorageSession(StorageSessionBase):
    def __init__(self, session_id, options=None):
        super(StorageSession, self).__init__(session_id=session_id)
        self._options = options if options else {}
        self._rp_session = None
        self._rpc = None

    def create(self):
        from eggroll.core.session import session_init
        from eggroll.roll_pair.roll_pair import RollPairContext
        self._options['eggroll.session.deploy.mode'] = "cluster"
        self._rp_session = session_init(session_id=self._session_id, options=self._options)
        self._rpc = RollPairContext(session=self._rp_session)
        self._session_id = self._rp_session.get_session_id()

    def table(self, name, namespace, address: AddressABC, partitions, storage_type: EggRollStorageType = EggRollStorageType.ROLLPAIR_LMDB, options=None, **kwargs):
        if isinstance(address, EggRollAddress):
            from fate_arch.storage.eggroll._table import StorageTable
            return StorageTable(context=self._rpc, name=name, namespace=namespace, address=address, partitions=partitions, storage_type=storage_type, options=options)
        raise NotImplementedError(f"address type {type(address)} not supported with eggroll storage")

    @log_elapsed
    def cleanup(self, name, namespace):
        self._rpc.cleanup(name=name, namespace=namespace)

    @log_elapsed
    def stop(self):
        return self._rp_session.stop()

    @log_elapsed
    def kill(self):
        return self._rp_session.kill()
