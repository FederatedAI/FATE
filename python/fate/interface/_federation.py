from typing import List, Optional, Protocol

from ._gc import GarbageCollector
from ._party import Parties, Party, PartyMeta


class FederationEngine(Protocol):
    session_id: str
    get_gc: Optional[GarbageCollector]
    remote_gc: Optional[GarbageCollector]
    local_party: PartyMeta
    parties: List[PartyMeta]

    def pull(self, name: str, tag: str, parties: List[PartyMeta]) -> List:
        ...

    def push(
        self,
        v,
        name: str,
        tag: str,
        parties: List[PartyMeta],
    ):
        ...

    def destroy(self, parties):
        ...


class FederationWrapper(Protocol):
    guest: Party
    hosts: Parties
    arbiter: Party
    parties: Parties


class FederationDeserializer(Protocol):
    def __do_deserialize__(self, ctx, party):
        ...
