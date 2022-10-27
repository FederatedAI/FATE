from typing import Any, List, Optional, Protocol, Tuple, TypeVar, overload

from ._consts import T_ROLE

T = TypeVar("T")


class _KeyedParty(Protocol):
    def put(self, value):
        ...

    def get(self) -> Any:
        ...


class Party(Protocol):
    def get(self, name: str) -> Any:
        ...

    @overload
    def put(self, name: str, value):
        ...

    @overload
    def put(self, **kwargs):
        ...

    def __call__(self, key: str) -> _KeyedParty:
        ...


class Parties(Protocol):
    def get(self, name: str) -> List:
        ...

    @overload
    def put(self, name: str, value):
        ...

    @overload
    def put(self, **kwargs):
        ...

    def __getitem__(self, key: int) -> Party:
        ...

    def get_neighbor(self, shift: int, module: bool = False) -> Party:
        ...

    def get_neighbors(self) -> "Parties":
        ...

    def get_local_index(self) -> Optional[int]:
        ...

    def __call__(self, key: str) -> _KeyedParty:
        ...


PartyMeta = Tuple[T_ROLE, str]
