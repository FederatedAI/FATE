from typing import Callable, List, Optional, Protocol, TypeVar

from ._tensor import FPTensor, PHEEncryptor, PHETensor

T = TypeVar("T")


class Future(Protocol):
    """
    `get` maybe async in future, in this version,
    we wrap obj to support explicit typing and check
    """

    def unwrap_tensor(self) -> "FPTensor":
        ...

    def unwrap_phe_encryptor(self) -> "PHEEncryptor":
        ...

    def unwrap_phe_tensor(self) -> "PHETensor":
        ...

    def unwrap(self, check: Optional[Callable[[T], bool]] = None) -> T:
        ...


class Futures(Protocol):
    def unwrap_tensors(self) -> List["FPTensor"]:
        ...

    def unwrap_phe_tensors(self) -> List["PHETensor"]:
        ...

    def unwrap(self, check: Optional[Callable[[T], bool]] = None) -> List[T]:
        ...


class Party(Protocol):
    def pull(self, name: str) -> Future:
        ...

    def push(self, name: str, value):
        ...


class Parties(Protocol):
    def pull(self, name: str) -> Futures:
        ...

    def push(self, name: str, value):
        ...

    def __call__(self, key: int) -> Party:
        ...
