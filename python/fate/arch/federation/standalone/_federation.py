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
from typing import List, Tuple

from fate.arch.abc import FederationEngine, PartyMeta

from ..._standalone import Federation as RawFederation
from ..._standalone import Table as RawTable
from ...computing.standalone import Table

LOGGER = logging.getLogger(__name__)


class StandaloneFederation(FederationEngine):
    def __init__(
        self,
        standalone_session,
        federation_session_id: str,
        party: PartyMeta,
        parties: List[PartyMeta],
    ):
        LOGGER.debug(
            f"[federation.standalone]init federation: "
            f"standalone_session={standalone_session}, "
            f"federation_session_id={federation_session_id}, "
            f"party={party}"
        )
        self._session_id = federation_session_id
        self._federation = RawFederation(standalone_session._session, federation_session_id, party)
        LOGGER.debug("[federation.standalone]init federation context done")
        self._remote_history = set()
        self._get_history = set()

        # standalone has build in design of table clean
        self.get_gc = None
        self.remote_gc = None
        self.local_party = party
        self.parties = parties

    @property
    def session_id(self) -> str:
        return self._session_id

    def push(
        self,
        v,
        name: str,
        tag: str,
        parties: List[Tuple[str, str]],
    ):
        for party in parties:
            if (name, tag, party) in self._remote_history:
                raise ValueError(f"remote to {parties} with duplicate tag: {name}.{tag}")
            self._remote_history.add((name, tag, party))

        if isinstance(v, Table):
            # noinspection PyProtectedMember
            v = v._table
        return self._federation.remote(v=v, name=name, tag=tag, parties=parties)

    def pull(
        self,
        name: str,
        tag: str,
        parties: List[Tuple[str, str]],
    ) -> List:
        for party in parties:
            if (name, tag, party) in self._get_history:
                raise ValueError(f"get from {party} with duplicate tag: {name}.{tag}")
            self._get_history.add((name, tag, party))
        rtn = self._federation.get(name=name, tag=tag, parties=parties)
        return [Table(r) if isinstance(r, RawTable) else r for r in rtn]

    def destroy(self):
        self._federation.destroy()
