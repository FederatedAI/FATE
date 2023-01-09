#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import abc
import typing
from abc import ABCMeta

from ..common import Party
from ._gc import GarbageCollectionABC

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
    def get(self, name: str, tag: str, parties: typing.List[Party], gc: GarbageCollectionABC) -> typing.List:
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
    def remote(
        self,
        v,
        name: str,
        tag: str,
        parties: typing.List[Party],
        gc: GarbageCollectionABC,
    ):
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
