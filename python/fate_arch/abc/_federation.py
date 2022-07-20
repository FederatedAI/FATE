import abc
import typing
from abc import ABCMeta

from fate_arch.abc._gc import GarbageCollectionABC
from fate_arch.common import Party

__all__ = ["FederationABC"]


class FederationABC(metaclass=ABCMeta):
    """
    federation, get or remote objects and tables
    """

    @property
    @abc.abstractmethod
    def session_id(self) -> str:
        ...

    @abc.abstractmethod
    def get(self, name: str,
            tag: str,
            parties: typing.List[Party],
            gc: GarbageCollectionABC) -> typing.List:
        """
        get objects/tables from ``parties``

        Parameters
        ----------
        name: str
           name of transfer variable
        tag: str
           tag to distinguish each transfer
        parties: typing.List[Party]
           parties to get objects/tables from
        gc: GarbageCollectionABC
           used to do some clean jobs

        Returns
        -------
        list
           a list of object or a list of table get from parties with same order of `parties`

        """
        ...

    @abc.abstractmethod
    def remote(self, v,
               name: str,
               tag: str,
               parties: typing.List[Party],
               gc: GarbageCollectionABC):
        """
        remote object/table to ``parties``

        Parameters
        ----------
        v: object or table
           object/table to remote
        name: str
           name of transfer variable
        tag: str
           tag to distinguish each transfer
        parties: typing.List[Party]
           parties to remote object/table to
        gc: GarbageCollectionABC
           used to do some clean jobs

        Returns
        -------
        Notes
        """
        ...

    @abc.abstractmethod
    def destroy(self, parties):
        """
        destroy federation from ``parties``

        Parameters
        ----------
        parties: typing.List[Party]
           parties to get objects/tables from

        Returns
        -------
        None
        """
        ...
