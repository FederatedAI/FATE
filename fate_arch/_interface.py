import abc
import typing
from typing_extensions import Protocol


class AddressABC(metaclass=abc.ABCMeta):
    ...


class GC(Protocol):

    def add_gc_func(self, tag: str, func: typing.Callable[[], typing.NoReturn]):
        ...
