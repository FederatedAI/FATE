from typing import List, Optional, Union

from fate_arch.common import Party
from fate_arch.federation.transfer_variable import FederationTagNamespace, IterationGC
from fate_arch.session import get_parties, get_session


def _get_role_parties(role: str):
    return get_parties().roles_to_parties([role], strict=False)


class _RoleIndexedParty:
    def __init__(self, role: str, index: int) -> None:
        assert index >= 0, "index should >= 0"
        self._role = role
        self._index = index

    @property
    def party(self) -> Party:
        parties = _get_role_parties(self._role)
        if 0 <= self._index < len(parties):
            return parties[self._index]
        raise KeyError(
            f"index `{self._index}` out of bound `0 <= index < {len(parties)}`"
        )


class _Parties:
    def __init__(
        self, parties: List[Union[str, Party, _RoleIndexedParty, "_Parties"]]
    ) -> None:
        self._parties = parties

    def _reverse(self):
        self._parties.reverse()
        return self

    @property
    def parties(self) -> List[Party]:
        flatten = []
        for p in self._parties:
            if isinstance(p, str) and (p == "guest" or p == "host" or p == "arbiter"):
                flatten.extend(_get_role_parties(p))
            elif isinstance(p, Party):
                flatten.append(p)
            elif isinstance(p, _RoleIndexedParty):
                flatten.append(p.party)
            elif isinstance(p, _Parties):
                flatten.extend(p.parties)
        return flatten

    def __add__(self, other) -> "_Parties":
        if isinstance(other, Party):
            return _Parties([self, other])
        elif isinstance(other, list):
            return _Parties([self, *other])
        else:
            raise ValueError(f"can't add `{other}`")

    def __radd__(self, other) -> "_Parties":
        return self.__add__(other)._reverse()


class _Role(_Parties):
    def __init__(self, role: str) -> None:
        self._role = role
        super().__init__([role])

    def __getitem__(self, key) -> _RoleIndexedParty:
        return _RoleIndexedParty(self._role, key)


ARBITER = _Role("arbiter")
GUEST = _Role("guest")
HOST = _Role("host")


class FedIter:
    def __init__(self, start_iter_num=-1) -> None:
        self._iter_num = start_iter_num
        self._is_start = False
        self._is_end = False
        self._push_gc_dict = {}
        self._pull_gc_dict = {}

    def __iter__(self):
        self.start()
        return self

    def __next__(self) -> int:
        self.increse_iter()
        return self._iter_num

    def start(self):
        self._is_start = True
        self._is_end = False

    def end(self):
        self._is_end = True

    def generate_iteration_aware_tag(self, name: str):
        if self._is_start:
            return f"{self._iter_num}.{name}"
        else:
            return name

    def increse_iter(self):
        self._iter_num += 1

    def push(self, target: _Parties, name: str, value):
        self._push(target.parties, name, value)
        return self

    def pull(self, source: _Parties, name: str) -> List:
        return self._pull(source.parties, name)

    def pull_one(self, source: _Parties, name: str):
        return self._pull(source.parties, name)[0]

    def _push(self, parties: List[Party], name, value):
        session = get_session()
        if self._is_start:
            name = self.generate_iteration_aware_tag(name)
        if name not in self._push_gc_dict:
            self._pull_gc_dict[name] = IterationGC()
        session.federation.remote(
            v=value,
            name=name,
            tag=FederationTagNamespace.generate_tag((name,)),
            parties=parties,
            gc=self._push_gc_dict[name],
        )

    def _pull(self, parties: List[Party], name):
        session = get_session()
        if self._is_start:
            name = self.generate_iteration_aware_tag(name)
        if name not in self._pull_gc_dict:
            self._pull_gc_dict[name] = IterationGC()
        return session.federation.get(
            name=name,
            tag=FederationTagNamespace.generate_tag((name,)),
            parties=parties,
            gc=self._pull_gc_dict[name],
        )

    def declare_key(self, name: str) -> "FedKey":
        return FedKey(name, self)


class FedKey:
    def __init__(self, name: str, fed_iter: Optional[FedIter] = None) -> None:
        self._name = name
        self._fed_iter = fed_iter

    def set_iter(self, fed_iter):
        self._fed_iter = fed_iter

    @property
    def name(self):
        return self._get_iter().generate_iteration_aware_tag(self._name)

    def _get_iter(self):
        if self._fed_iter is None:
            raise RuntimeError(f"fed_iter not set")
        return self._fed_iter

    def push(self, target: _Parties, value):
        self._get_iter()._push(target.parties, self.name, value)
        return self

    def pull(self, source: _Parties) -> List:
        return self._get_iter()._pull(source.parties, self.name)

    def pull_one(self, source: _Parties):
        return self._get_iter()._pull(source.parties, self.name)[0]
