from typing import Protocol

from fate.arch.unify import device

from ._dtype import dtype
from ._shape import Shape


class LStorage(Protocol):
    device: device
    dtype: dtype
    shape: Shape

    def tolist(self):
        ...

    def transpose(self) -> "LStorage":
        ...

    def to_local(self) -> "LStorage":
        ...
