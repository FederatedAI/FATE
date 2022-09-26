from typing import List, Literal, Optional, Tuple

from fate.interface import FederationEngine, PartyMeta

from ..._standalone import Federation as RawFederation
from ..._standalone import Table as RawTable
from ...common import log
from ...computing.standalone import Table

LOGGER = log.getLogger()


class StandaloneFederation(FederationEngine):
    def __init__(self, standalone_session, federation_session_id: str, party: PartyMeta, parties: List[PartyMeta]):
        LOGGER.debug(
            f"[federation.standalone]init federation: "
            f"standalone_session={standalone_session}, "
            f"federation_session_id={federation_session_id}, "
            f"party={party}"
        )
        self._session_id = federation_session_id
        self._federation = RawFederation(standalone_session, federation_session_id, party)
        LOGGER.debug("[federation.standalone]init federation context done")
        self._remote_history = set()
        self._get_history = set()

        # standalone has build in design of table clean
        self.get_gc = None
        self.remote_gc = None
        self.party = party
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

    def destroy(self, parties):
        self._federation.destroy()
