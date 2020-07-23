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


import typing

from eggroll.core.session import session_init
from eggroll.roll_pair.roll_pair import RollPairContext
from eggroll.utils import file_utils
from fate_arch._interface import AddressABC
from fate_arch.common import WorkMode, Party
from fate_arch.common.log import getLogger
from fate_arch.common.profile import log_elapsed
from fate_arch.session._interface import SessionABC
from fate_arch.session._session_types import _FederationParties
from fate_arch.session.impl._file import Path

from fate_arch.session.impl.eggroll._federation import FederationEngine
from fate_arch.session.impl.eggroll._table import Table

LOGGER = getLogger()


class Session(SessionABC):
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

        self._federation_session: typing.Optional[FederationEngine] = None
        self._federation_parties: typing.Optional[_FederationParties] = None

    def _init_federation(self, federation_session_id: str,
                         party: Party,
                         parties: typing.MutableMapping[str, typing.List[Party]],
                         host: str, port: int):
        if self._federation_session is not None:
            raise RuntimeError("federation session already initialized")
        self._federation_session = FederationEngine(self._rpc, federation_session_id, party, host, int(port))
        self._federation_parties = _FederationParties(party, parties)

    def init_federation(self, federation_session_id: str, runtime_conf: dict, server_conf: typing.Optional[str] = None):
        if server_conf is None:
            _path = file_utils.get_project_base_directory() + "/arch/conf/server_conf.json"
            server_conf = file_utils.load_json_conf(_path)
        host = server_conf.get('servers').get('proxy').get("host")
        port = server_conf.get('servers').get('proxy').get("port")

        party, parties = self._parse_runtime_conf(runtime_conf)
        self._init_federation(federation_session_id, party, parties, host, int(port))

    def _get_session_id(self):
        return self._session_id

    def _get_federation(self):
        return self._federation_session

    def _get_federation_parties(self):
        return self._federation_parties

    @log_elapsed
    def load(self, address: AddressABC, partitions: int, schema: dict, **kwargs) -> typing.Union[Table, Path]:

        from fate_arch.data_table.base import EggRollAddress
        if isinstance(address, EggRollAddress):
            options = kwargs.get("option", {})
            options["total_partitions"] = partitions
            rp = self._rpc.load(namespace=address.namespace, name=address.name, options=options)
            table = Table(rp=rp)
            table.schema = schema
            return table

        from fate_arch.data_table.base import FileAddress
        if isinstance(address, FileAddress):
            return Path(address.path, address.path_type)

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
