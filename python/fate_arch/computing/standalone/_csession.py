#
#  Copyright 2019 The Eggroll Authors. All Rights Reserved.
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
from collections import Iterable

from fate_arch._standalone import Session
from fate_arch.abc import AddressABC, CSessionABC
from fate_arch.common.base_utils import fate_uuid
from fate_arch.common.log import getLogger
from fate_arch.computing.standalone._table import Table

LOGGER = getLogger()


class CSession(CSessionABC):
    def __init__(self, session_id: str, options=None):
        if options is not None:
            max_workers = options.get("task_cores", None)
        self._session = Session(session_id, max_workers=max_workers)

    def get_standalone_session(self):
        return self._session

    def load(self, address: AddressABC, partitions: int, schema: dict, **kwargs):
        from fate_arch.common.address import StandaloneAddress
        from fate_arch.storage import StandaloneStoreType

        if isinstance(address, StandaloneAddress):
            raw_table = self._session.load(address.name, address.namespace)
            if address.storage_type != StandaloneStoreType.ROLLPAIR_IN_MEMORY:
                raw_table = raw_table.save_as(
                    name=f"{address.name}_{fate_uuid()}",
                    namespace=address.namespace,
                    partition=partitions,
                    need_cleanup=True,
                )
            table = Table(raw_table)
            table.schema = schema
            return table

        from fate_arch.common.address import PathAddress

        if isinstance(address, PathAddress):
            from fate_arch.computing.non_distributed import LocalData
            from fate_arch.computing import ComputingEngine
            return LocalData(address.path, engine=ComputingEngine.STANDALONE)
        raise NotImplementedError(
            f"address type {type(address)} not supported with standalone backend"
        )

    def parallelize(self, data: Iterable, partition: int, include_key: bool, **kwargs):
        table = self._session.parallelize(
            data=data, partition=partition, include_key=include_key, **kwargs
        )
        return Table(table)

    def cleanup(self, name, namespace):
        return self._session.cleanup(name=name, namespace=namespace)

    def stop(self):
        return self._session.stop()

    def kill(self):
        return self._session.kill()

    @property
    def session_id(self):
        return self._session.session_id
