import enum
from typing import List

from fate.arch.common import Party
from fate.arch.session import get_parties


class Parties:
    def __init__(self, flag) -> None:
        self._flag = flag

    def contains_hosts(self) -> bool:
        return bool(self._flag & 1)

    def contains_arbiter(self) -> bool:
        return bool(self._flag & 2)

    def contains_guest(self) -> bool:
        return bool(self._flag & 4)

    def contains_host(self) -> bool:
        return bool(self._flag & 8)

    @property
    def indexes(self) -> List[int]:
        return [i for i, e in enumerate(bin(self._flag)[::-1]) if e == "1"]

    @classmethod
    def get_name(cls, i):
        if i < 4:
            return {0: "HOSTS", 1: "ARBITER", 2: "GUEST", 3: "HOST"}[i]
        else:
            return f"HOST{i-3}"

    def __or__(self, other):
        return Parties(self._flag | other._flag)

    def __ror__(self, other):
        return Parties(self._flag | other._flag)

    def __hash__(self) -> int:
        return self._flag

    def __eq__(self, o) -> bool:
        return self._flag == o._flag

    def __str__(self):
        readable = "|".join([self.get_name(i) for i in self.indexes])
        return f"<Parties({self._flag:0>4b}): {readable}>"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, key):
        if self._flag == 1 and isinstance(key, int) and key >= 0:
            return Parties(1 << (key + 3))
        raise TypeError("not subscriptable")

    def _get_role_parties(self, role: str):
        return get_parties().roles_to_parties([role], strict=False)

    def get_parties(self) -> List[Party]:
        parties = []
        if self._flag & 2:
            parties.extend(self._get_role_parties("arbiter"))
        if self._flag & 4:
            parties.extend(self._get_role_parties("guest"))
        if self._flag & 1:
            parties.extend(self._get_role_parties("host"))
        else:
            host_bit_int = self._flag >> 3
            if host_bit_int:
                hosts = self._get_role_parties("host")
                for i, e in enumerate(bin(host_bit_int)[::-1]):
                    if e == "1":
                        parties.append(hosts[i])
        return parties


class PreludeParty(Parties, enum.Flag):
    HOSTS = 1
    ARBITER = 2
    GUEST = 4
    HOST = 8
