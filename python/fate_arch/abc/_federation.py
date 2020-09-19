import abc
import typing
from abc import ABCMeta

from fate_arch.abc._gc import GarbageCollectionABC
from fate_arch.common import Party


class FederationABC(metaclass=ABCMeta):

    @abc.abstractmethod
    def get(self, name: str,
            tag: str,
            parties: typing.List[Party],
            gc: GarbageCollectionABC) -> typing.List:
        ...

    @abc.abstractmethod
    def remote(self, v,
               name: str,
               tag: str,
               parties: typing.List[Party],
               gc: GarbageCollectionABC) -> typing.NoReturn:
        ...
