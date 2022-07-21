from typing import List, Union

from fate_arch.common import Party
from fate_arch.session import get_parties


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
