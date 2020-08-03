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
import typing
from collections import Iterable

from fate_arch.abc import AddressABC, CSessionABC
from fate_arch.backend.standalone import Session as RawSession
from fate_arch.common import Party
from fate_arch.common.log import getLogger
from fate_arch.session._parties_util import _FederationParties
from fate_arch.session.standalone._federation import Federation
from fate_arch.session.standalone._table import Table

LOGGER = getLogger()


class Session(CSessionABC):

    def __init__(self, session: RawSession):
        self._session = session
        self._federation_session: typing.Optional[Federation] = None
        self._federation_parties: typing.Optional[_FederationParties] = None

    def _init_federation(self, federation_session_id: str,
                         party: Party,
                         parties: typing.MutableMapping[str, typing.List[Party]]):
        if self._federation_session is not None:
            raise RuntimeError("federation session already initialized")
        self._federation_session = Federation(federation_session_id, party)
        self._federation_parties = _FederationParties(party, parties)

    def init_federation(self, federation_session_id: str, runtime_conf: dict, **kwargs):
        party, parties = self._parse_runtime_conf(runtime_conf)
        self._init_federation(federation_session_id, party, parties)

    def load(self, address: AddressABC, partitions: int, schema: dict, **kwargs):
        from fate_arch.data_table.address import EggRollAddress
        if isinstance(address, EggRollAddress):
            table = Table(self._session.load(address.name, address.namespace))
            table.schema = schema
            return table

        from fate_arch.data_table.address import FileAddress
        if isinstance(address, FileAddress):
            from fate_arch.session._file import Path as _Path
            return _Path(address.path, address.path_type)
        raise NotImplementedError(f"address type {type(address)} not supported with standalone backend")

    def parallelize(self, data: Iterable, partition: int, include_key: bool = False, **kwargs):
        table = self._session.parallelize(data=data, partition=partition, include_key=include_key, **kwargs)
        return Table(table)

    def cleanup(self, name, namespace):
        return self._session.cleanup(name=name, namespace=namespace)

    def stop(self):
        return self._session.stop()

    def kill(self):
        return self._session.kill()

    def _get_federation(self):
        return self._federation_session

    def _get_session_id(self):
        return self._session.session_id

    def _get_federation_parties(self):
        raise self._federation_parties
