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


from eggroll.core.session import session_init
from eggroll.roll_pair.roll_pair import runtime_init

from fate_arch.abc import AddressABC, CSessionABC
from fate_arch.common.base_utils import fate_uuid
from fate_arch.common.log import getLogger
from fate_arch.common.profile import computing_profile
from fate_arch.computing.eggroll import Table

LOGGER = getLogger()


class CSession(CSessionABC):
    def __init__(self, session_id, options: dict = None):
        if options is None:
            options = {}
        if "eggroll.session.deploy.mode" not in options:
            options["eggroll.session.deploy.mode"] = "cluster"
        if "eggroll.rollpair.inmemory_output" not in options:
            options["eggroll.rollpair.inmemory_output"] = True
        self._rp_session = session_init(session_id=session_id, options=options)
        self._rpc = runtime_init(session=self._rp_session)
        self._session_id = self._rp_session.get_session_id()

    def get_rpc(self):
        return self._rpc

    @property
    def session_id(self):
        return self._session_id

    @computing_profile
    def load(self, address: AddressABC, partitions: int, schema: dict, **kwargs):

        from fate_arch.common.address import EggRollAddress
        from fate_arch.storage import EggRollStoreType

        if isinstance(address, EggRollAddress):
            options = kwargs.get("option", {})
            options["total_partitions"] = partitions
            options["store_type"] = kwargs.get("store_type", EggRollStoreType.ROLLPAIR_LMDB)
            options["create_if_missing"] = False
            rp = self._rpc.load(
                namespace=address.namespace, name=address.name, options=options
            )
            if rp is None or rp.get_partitions() == 0:
                raise RuntimeError(
                    f"no exists: {address.name}, {address.namespace}"
                )

            if options["store_type"] != EggRollStoreType.ROLLPAIR_IN_MEMORY:
                rp = rp.save_as(
                    name=f"{address.name}_{fate_uuid()}",
                    namespace=self.session_id,
                    partition=partitions,
                    options={"store_type": EggRollStoreType.ROLLPAIR_IN_MEMORY},
                )

            table = Table(rp=rp)
            table.schema = schema
            return table

        from fate_arch.common.address import PathAddress

        if isinstance(address, PathAddress):
            from fate_arch.computing.non_distributed import LocalData
            from fate_arch.computing import ComputingEngine
            return LocalData(address.path, engine=ComputingEngine.EGGROLL)

        raise NotImplementedError(
            f"address type {type(address)} not supported with eggroll backend"
        )

    @computing_profile
    def parallelize(self, data, partition: int, include_key: bool, **kwargs) -> Table:
        options = dict()
        options["total_partitions"] = partition
        options["include_key"] = include_key
        rp = self._rpc.parallelize(data=data, options=options)
        return Table(rp)

    def cleanup(self, name, namespace):
        self._rpc.cleanup(name=name, namespace=namespace)

    def stop(self):
        return self._rp_session.stop()

    def kill(self):
        return self._rp_session.kill()
