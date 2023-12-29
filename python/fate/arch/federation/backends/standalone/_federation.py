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
import logging
from typing import List

from fate.arch.computing.backends.standalone import Table, CSession
from fate.arch.computing.backends.standalone import standalone_raw
from fate.arch.federation.api import Federation, TableMeta, PartyMeta

LOGGER = logging.getLogger(__name__)


class StandaloneFederation(Federation):
    def __init__(
        self,
        standalone_session: CSession,
        federation_session_id: str,
        party: PartyMeta,
        parties: List[PartyMeta],
    ):
        super().__init__(federation_session_id, party, parties)
        self._federation = standalone_raw.Federation.create(
            standalone_session.get_standalone_session(), session_id=federation_session_id, party=party
        )

    def _push_table(
        self,
        table: Table,
        name: str,
        tag: str,
        parties: List[PartyMeta],
    ):
        return self._federation.push_table(table=table.table, name=name, tag=tag, parties=parties)

    def _push_bytes(
        self,
        v: bytes,
        name: str,
        tag: str,
        parties: List[PartyMeta],
    ):
        return self._federation.push_bytes(v=v, name=name, tag=tag, parties=parties)

    def _pull_table(self, name: str, tag: str, parties: List[PartyMeta], table_metas: List[TableMeta]) -> List[Table]:
        rtn = self._federation.pull_table(name=name, tag=tag, parties=parties)

        return [Table(r) if isinstance(r, standalone_raw.Table) else r for r in rtn]

    def _pull_bytes(
        self,
        name: str,
        tag: str,
        parties: List[PartyMeta],
    ) -> List[bytes]:
        rtn = self._federation.pull_bytes(name=name, tag=tag, parties=parties)

        return [Table(r) if isinstance(r, standalone_raw.Table) else r for r in rtn]

    def _destroy(self):
        self._federation.destroy()
