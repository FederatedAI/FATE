from typing import Callable, List, Optional, TypeVar

from fate.interface import FederationEngine, FPTensor
from fate.interface import Future as FutureInterface
from fate.interface import Futures as FuturesInterface
from fate.interface import Parties as PartiesInterface
from fate.interface import Party as PartyInterface
from fate.interface import PartyMeta, PHEEncryptor, PHETensor

from ..federation.transfer_variable import IterationGC
from ._namespace import Namespace

T = TypeVar("T")


class Future(FutureInterface):
    def __init__(self, inside) -> None:
        self._inside = inside

    def unwrap_tensor(self) -> "FPTensor":

        assert isinstance(self._inside, FPTensor)
        return self._inside

    def unwrap_phe_encryptor(self) -> "PHEEncryptor":
        assert isinstance(self._inside, PHEEncryptor)
        return self._inside

    def unwrap_phe_tensor(self) -> "PHETensor":

        assert isinstance(self._inside, PHETensor)
        return self._inside

    def unwrap(self, check: Optional[Callable[[T], bool]] = None) -> T:
        if check is not None and not check(self._inside):
            raise TypeError(f"`{self._inside}` check failed")
        return self._inside


class Futures(FuturesInterface):
    def __init__(self, insides) -> None:
        self._insides = insides

    def unwrap_tensors(self) -> List["FPTensor"]:

        for t in self._insides:
            assert isinstance(t, FPTensor)
        return self._insides

    def unwrap_phe_tensors(self) -> List["PHETensor"]:

        for t in self._insides:
            assert isinstance(t, PHETensor)
        return self._insides

    def unwrap(self, check: Optional[Callable[[T], bool]] = None) -> List[T]:
        if check is not None:
            for i, t in enumerate(self._insides):
                if not check(t):
                    raise TypeError(f"{i}th element `{self._insides}` check failed")
        return self._insides


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


class Party(PartyInterface):
    def __init__(
        self,
        federation,
        party: PartyMeta,
        namespace,
    ) -> None:
        self.federation = federation
        self.party = party
        self.namespace = namespace

    def push(self, name: str, value):
        return _push(
            self.federation,
            name,
            self.namespace,
            [self.party],
            value,
        )

    def pull(self, name: str) -> Future:
        return Future(_pull(self.federation, name, self.namespace, [self.party])[0])


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

    def __call__(self, key: int) -> Party:
        return Party(
            self.federation,
            self.parties[key],
            self.namespace,
        )

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
        return Parties(self.ctx, self.federation, self.party, parties, self.namespace)

    def get_local_index(self) -> Optional[int]:
        if self.party not in self.parties:
            return None
        else:
            return self.parties.index(self.party)

    def push(self, name: str, value):
        return _push(
            self.federation,
            name,
            self.namespace,
            self.parties,
            value,
        )

    def pull(self, name: str) -> Futures:
        return Futures(_pull(self.federation, name, self.namespace, self.parties))


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
