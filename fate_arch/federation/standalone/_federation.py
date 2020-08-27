import typing

from fate_arch.abc import FederationABC
from fate_arch.abc import GarbageCollectionABC
from fate_arch.common import Party
from fate_arch.computing.standalone import Table
from fate_arch.standalone import Federation as RawFederation, Table as RawTable


class Federation(FederationABC):

    def __init__(self, standalone_session, federation_session_id, party):
        self._federation = RawFederation(standalone_session, federation_session_id, party)

    def remote(self, v, name: str, tag: str, parties: typing.List[Party], gc: GarbageCollectionABC):
        if isinstance(v, Table):
            v = v.as_federation_format()
        return self._federation.remote(v=v, name=name, tag=tag, parties=parties)

    # noinspection PyProtectedMember
    def get(self, name: str, tag: str, parties: typing.List[Party], gc: GarbageCollectionABC) -> typing.List:
        rtn = self._federation.get(name=name, tag=tag, parties=parties)
        return [Table(r) if isinstance(r, RawTable) else r for r in rtn]
