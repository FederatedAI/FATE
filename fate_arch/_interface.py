import abc
import typing
from typing_extensions import Protocol


class AddressABC(metaclass=abc.ABCMeta):
    ...


class GC(Protocol):

    def add_gc_action(self, tag: str, obj, method, args_dict):
        ...
