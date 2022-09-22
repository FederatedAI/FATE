from typing import Callable, List, Literal, Optional, Tuple, TypeVar

from fate.interface import FPTensor
from fate.interface import Future as FutureInterface
from fate.interface import Futures as FuturesInterface
from fate.interface import Parties as PartiesInterface
from fate.interface import Party as PartyInterface
from fate.interface import PHEEncryptor, PHETensor
from fate.interface import FederationEngine as FederationEngineInterface

from ..common import Party as PartyMeta
from ..federation.transfer_variable import IterationGC
from ._namespace import Namespace


class FederationDeserializer:
    def do_deserialize(self, ctx, party):
        ...

    @classmethod
    def make_frac_key(cls, base_key, frac_key):
        return f"{base_key}__frac__{frac_key}"


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


class FederationParty(PartyInterface):
    def __init__(
        self, ctx, federation, party: Tuple[str, str], namespace, gc: GC
    ) -> None:
        self.ctx = ctx
        self.federation = federation
        self.party = PartyMeta(party[0], party[1])
        self.namespace = namespace
        self.gc = gc

    def push(self, name: str, value):
        return _push(
            self.ctx,
            self.federation,
            name,
            self.namespace,
            [self.party],
            self.gc,
            value,
        )

    def pull(self, name: str) -> Future:
        return Future(
            _pull(
                self.ctx, self.federation, name, self.namespace, [self.party], self.gc
            )[0]
        )


class FederationParties(PartiesInterface):
    def __init__(
        self,
        ctx,
        federation,
        parties: List[Tuple[str, str]],
        namespace: Namespace,
        gc: GC,
    ) -> None:
        self.ctx = ctx
        self.federation = federation
        self.parties = [PartyMeta(party[0], party[1]) for party in parties]
        self.namespace = namespace
        self.gc = gc

    def __call__(self, key: int) -> FederationParty:
        return FederationParty(
            self.ctx,
            self.federation,
            self.parties[key].as_tuple(),
            self.namespace,
            self.gc,
        )

    def push(self, name: str, value):
        return _push(
            self.ctx,
            self.federation,
            name,
            self.namespace,
            self.parties,
            self.gc,
            value,
        )

    def pull(self, name: str) -> Futures:
        return Futures(
            _pull(
                self.ctx, self.federation, name, self.namespace, self.parties, self.gc
            )
        )


def _push(
    ctx,
    federation,
    name: str,
    namespace: Namespace,
    parties: List[PartyMeta],
    gc: GC,
    value,
):
    if hasattr(value, "__federation_hook__"):
        value.__federation_hook__(ctx, name, parties)
    else:
        federation.remote(
            v=value,
            name=name,
            tag=namespace.fedeation_tag(),
            parties=parties,
            gc=gc.get_or_set_push_gc(name),
        )


def _pull(
    ctx, federation, name: str, namespace: Namespace, parties: List[PartyMeta], gc: GC
):
    raw_values = federation.get(
        name=name,
        tag=namespace.fedeation_tag(),
        parties=parties,
        gc=gc.get_or_set_pull_gc(name),
    )
    values = []
    for party, raw_value in zip(parties, raw_values):
        if isinstance(raw_value, FederationDeserializer):
            values.append(raw_value.do_deserialize(ctx, party))
        else:
            values.append(raw_value)
    return values


class FederationEngine(FederationEngineInterface):
    def __init__(
        self,
        federation_id: str,
        local_party: Tuple[Literal["guest", "host", "arbiter"], str],
        parties: Optional[List[Tuple[Literal["guest", "host", "arbiter"], str]]],
        ctx,
        session,  # should remove
        namespace: Namespace,
    ):
        if parties is None:
            parties = []
        if local_party not in parties:
            parties.append(local_party)
        self._local = local_party
        self._parties = parties
        self._role_to_parties = {}
        for (role, party_id) in self._parties:
            self._role_to_parties.setdefault(role, []).append(party_id)

        # walkround, temp
        from ..common._parties import Party, PartiesInfo

        local = Party(local_party[0], local_party[1])
        role_to_parties = {}
        for role, party_id in [local_party, *parties]:
            role_to_parties.setdefault(role, []).append(Party(role, party_id))
        session.init_federation(
            federation_session_id=federation_id,
            parties_info=PartiesInfo(local, role_to_parties),
        )
        self.federation = session.federation

        self.ctx = ctx
        self.namespace = namespace
        self.gc = GC()

    @property
    def guest(self) -> PartyInterface:
        party = self._role("guest")[0]
        return FederationParty(
            self.ctx, self.federation, party, self.namespace, self.gc
        )

    @property
    def hosts(self) -> PartiesInterface:
        parties = self._role("host")
        return FederationParties(
            self.ctx, self.federation, parties, self.namespace, self.gc
        )

    @property
    def arbiter(self) -> PartyInterface:
        party = self._role("arbiter")[0]
        return FederationParty(
            self.ctx, self.federation, party, self.namespace, self.gc
        )

    @property
    def parties(self) -> PartiesInterface:
        return FederationParties(
            self.ctx, self.federation, self._parties, self.namespace, self.gc
        )

    def _role(self, role: str) -> List:
        if role not in self._role_to_parties:
            raise RuntimeError(f"no {role} party has configurated")
        return [(role, party_id) for party_id in self._role_to_parties[role]]
