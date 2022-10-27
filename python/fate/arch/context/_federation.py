from typing import List, Optional, TypeVar, Union

from fate.interface import FederationEngine
from fate.interface import Parties as PartiesInterface
from fate.interface import Party as PartyInterface
from fate.interface import PartyMeta

from ..federation.transfer_variable import IterationGC
from ._namespace import Namespace

T = TypeVar("T")


class GC:
    def __init__(self) -> None:
        self._push_gc_dict = {}
        self._pull_gc_dict = {}

    def get_or_set_push_gc(self, key):
        if key not in self._push_gc_dict:
            self._push_gc_dict[key] = IterationGC()
        return self._push_gc_dict[key]

    def get_or_set_pull_gc(self, key):
        if key not in self._pull_gc_dict:
            self._pull_gc_dict[key] = IterationGC()
        return self._pull_gc_dict[key]


class _KeyedParty:
    def __init__(self, party: Union["Party", "Parties"], key) -> None:
        self.party = party
        self.key = key

    def put(self, value):
        return self.party.put(self.key, value)

    def get(self):
        return self.party.get(self.key)


class Party(PartyInterface):
    def __init__(self, federation, party: PartyMeta, namespace, key=None) -> None:
        self.federation = federation
        self.party = party
        self.namespace = namespace
        self.key = key

    def __call__(self, key: str) -> "_KeyedParty":
        return _KeyedParty(self, key)

    def put(self, *args, **kwargs):
        if args:
            assert len(args) == 2 and isinstance(
                args[0], str
            ), "invalid position parameter"
            assert (
                not kwargs
            ), "keywords paramters not allowed when position parameter provided"
            kvs = [args]
        else:
            kvs = kwargs.items()

        for k, v in kvs:
            return _push(self.federation, k, self.namespace, [self.party], v)

    def get(self, name: str):
        return _pull(self.federation, name, self.namespace, [self.party])[0]


class Parties(PartiesInterface):
    def __init__(
        self,
        federation: FederationEngine,
        party: PartyMeta,
        parties: List[PartyMeta],
        namespace: Namespace,
    ) -> None:
        self.federation = federation
        self.party = party
        self.parties = parties
        self.namespace = namespace

    def __getitem__(self, key: int) -> Party:
        return Party(self.federation, self.parties[key], self.namespace)

    def __call__(self, key: str) -> "_KeyedParty":
        return _KeyedParty(self, key)

    def get_neighbor(self, shift: int, module: bool = False) -> Party:
        start_index = self.get_local_index()
        if start_index is None:
            raise RuntimeError(f"local party `{self.party}` not in `{self.parties}`")
        target_index = start_index + shift
        if module:
            target_index = target_index % module

        if 0 <= target_index < len(self.parties):
            return self(target_index)
        else:
            raise IndexError(f"target index `{target_index}` out of bound")

    def get_neighbors(self) -> "Parties":
        parties = [party for party in self.parties if party != self.party]
        return Parties(self.federation, self.party, parties, self.namespace)

    def get_local_index(self) -> Optional[int]:
        if self.party not in self.parties:
            return None
        else:
            return self.parties.index(self.party)

    def put(self, *args, **kwargs):
        if args:
            assert len(args) == 2 and isinstance(
                args[0], str
            ), "invalid position parameter"
            assert (
                not kwargs
            ), "keywords paramters not allowed when position parameter provided"
            kvs = [args]
        else:
            kvs = kwargs.items()
        for k, v in kvs:
            return _push(self.federation, k, self.namespace, self.parties, v)

    def get(self, name: str):
        return _pull(self.federation, name, self.namespace, self.parties)


def _push(
    federation: FederationEngine,
    name: str,
    namespace: Namespace,
    parties: List[PartyMeta],
    value,
):
    if hasattr(value, "__federation_hook__"):
        value.__federation_hook__(federation, name, namespace.fedeation_tag(), parties)
    else:
        federation.push(
            v=value,
            name=name,
            tag=namespace.fedeation_tag(),
            parties=parties,
        )


def _pull(
    federation: FederationEngine,
    name: str,
    namespace: Namespace,
    parties: List[PartyMeta],
):
    tag = namespace.fedeation_tag()
    raw_values = federation.pull(
        name=name,
        tag=tag,
        parties=parties,
    )
    values = []
    for party, raw_value in zip(parties, raw_values):
        if hasattr(raw_value, "__do_deserialize__"):
            values.append(raw_value.__do_deserialize__(federation, name, tag, party))
        else:
            values.append(raw_value)
    return values
