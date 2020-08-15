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


from eggroll.core.constants import StoreTypes
from eggroll.core.session import session_init
from eggroll.roll_pair.roll_pair import RollPairContext
from fate_arch.abc import AddressABC, CSessionABC
from fate_arch.common import WorkMode
from fate_arch.common.log import getLogger
from fate_arch.common.profile import log_elapsed
from fate_arch.computing.eggroll import Table
from fate_arch.common.base_utils import fate_uuid

LOGGER = getLogger()


class CSession(CSessionABC):
    def __init__(self, session_id, work_mode, options: dict = None):
        if options is None:
            options = {}
        if work_mode == WorkMode.STANDALONE:
            options['eggroll.session.deploy.mode'] = "standalone"
        elif work_mode == WorkMode.CLUSTER:
            options['eggroll.session.deploy.mode'] = "cluster"
        self._rp_session = session_init(session_id=session_id, options=options)
        self._rpc = RollPairContext(session=self._rp_session)
        self._session_id = self._rp_session.get_session_id()
        self._default_storage_type = options.get("store_type", StoreTypes.ROLLPAIR_IN_MEMORY)

    def get_rpc(self):
        return self._rpc

    @property
    def session_id(self):
        return self._session_id

    @log_elapsed
    def load(self, address: AddressABC, partitions: int, schema: dict, **kwargs):

        from fate_arch.common.address import EggRollAddress
        if isinstance(address, EggRollAddress):
            options = kwargs.get("option", {})
            options["total_partitions"] = partitions
            options["store_type"] = address.storage_type
            options["create_if_missing"] = False
            rp = self._rpc.load(namespace=address.namespace, name=address.name, options=options)
            if rp is None or rp.get_partitions() == 0:
                raise RuntimeError(f"no exists: {address.name}, {address.namespace}, {address.storage_type}")

            if address.storage_type != StoreTypes.ROLLPAIR_IN_MEMORY:
                rp = rp.save_as(name=f"{address.name}_{fate_uuid()}", namespace=address.namespace, partition=partitions,
                                options={'store_type': StoreTypes.ROLLPAIR_IN_MEMORY})

            table = Table(rp=rp)
            table.schema = schema
            return table

        from fate_arch.common.address import FileAddress
        if isinstance(address, FileAddress):
            return address

        raise NotImplementedError(f"address type {type(address)} not supported with eggroll backend")

    @log_elapsed
    def parallelize(self, data, partition: int, include_key: bool, **kwargs) -> Table:
        options = dict()
        options["total_partitions"] = partition
        options["include_key"] = include_key
        rp = self._rpc.parallelize(data=data, options=options)
        return Table(rp)

    @log_elapsed
    def cleanup(self, name, namespace):
        self._rpc.cleanup(name=name, namespace=namespace)

    @log_elapsed
    def stop(self):
        return self._rp_session.stop()

    @log_elapsed
    def kill(self):
        return self._rp_session.kill()
