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
#
import hashlib
import typing
from collections import deque
from typing import Union

from fate_arch.abc import GarbageCollectionABC
from fate_arch.common import Party, profile
from fate_arch.common.log import getLogger
from fate_arch.session import get_session, get_parties

__all__ = ["Variable", "BaseTransferVariables"]

LOGGER = getLogger()


class FederationTagNamespace(object):
    __namespace = "default"

    @classmethod
    def set_namespace(cls, namespace):
        cls.__namespace = namespace

    @classmethod
    def generate_tag(cls, *suffix):
        tags = (cls.__namespace, *map(str, suffix))
        return ".".join(tags)


class IterationGC(GarbageCollectionABC):
    def __init__(self, capacity=2):
        self._ashcan: deque[typing.List[typing.Tuple[typing.Any, str, dict]]] = deque()
        self._last_tag: typing.Optional[str] = None
        self._capacity = capacity
        self._enable = True

    def add_gc_action(self, tag: str, obj, method, args_dict):
        if self._last_tag == tag:
            self._ashcan[-1].append((obj, method, args_dict))
        else:
            self._ashcan.append([(obj, method, args_dict)])
            self._last_tag = tag

    def disable(self):
        self._enable = False

    def set_capacity(self, capacity):
        self._capacity = capacity

    def gc(self):
        if not self._enable:
            return
        if len(self._ashcan) <= self._capacity:
            return
        self._safe_gc_call(self._ashcan.popleft())

    def clean(self):
        while self._ashcan:
            self._safe_gc_call(self._ashcan.pop())

    @staticmethod
    def _safe_gc_call(actions: typing.List[typing.Tuple[typing.Any, str, dict]]):
        for obj, method, args_dict in actions:
            try:
                LOGGER.debug(f"[CLEAN]deleting {obj}, {method}, {args_dict}")
                getattr(obj, method)(**args_dict)
            except Exception as e:
                LOGGER.debug(f"[CLEAN]this could be ignore {e}")


class Variable(object):
    """
    variable to distinguish federation by name
    """

    __instances: typing.MutableMapping[str, "Variable"] = {}

    @classmethod
    def get_or_create(
        cls, name, create_func: typing.Callable[[], "Variable"]
    ) -> "Variable":
        if name not in cls.__instances:
            value = create_func()
            cls.__instances[name] = value
        return cls.__instances[name]

    def __init__(
        self, name: str, src: typing.Tuple[str, ...], dst: typing.Tuple[str, ...]
    ):

        if name in self.__instances:
            raise RuntimeError(
                f"{self.__instances[name]} with {name} already initialized, which expected to be an singleton object."
            )

        assert (
            len(name.split(".")) >= 3
        ), "incorrect name format, should be `module_name.class_name.variable_name`"
        self._name = name
        self._src = src
        self._dst = dst
        self._get_gc = IterationGC()
        self._remote_gc = IterationGC()
        self._use_short_name = True
        self._short_name = self._get_short_name(self._name)

    @staticmethod
    def _get_short_name(name):
        fix_sized = hashlib.blake2b(name.encode("utf-8"), digest_size=10).hexdigest()
        _, right = name.rsplit(".", 1)
        return f"hash.{fix_sized}.{right}"

    # copy never create a new instance
    def __copy__(self):
        return self

    # deepcopy never create a new instance
    def __deepcopy__(self, memo):
        return self

    def set_preserve_num(self, n):
        self._get_gc.set_capacity(n)
        self._remote_gc.set_capacity(n)
        return self

    def disable_auto_clean(self):
        self._get_gc.disable()
        self._remote_gc.disable()
        return self

    def clean(self):
        self._get_gc.clean()
        self._remote_gc.clean()

    def remote_parties(
        self,
        obj,
        parties: Union[typing.List[Party], Party],
        suffix: Union[typing.Any, typing.Tuple] = tuple(),
    ):
        """
        remote object to specified parties

        Parameters
        ----------
        obj: object or table
           object or table to remote
        parties: typing.List[Party]
           parties to remote object/table to
        suffix: str or tuple of str
           suffix used to distinguish federation with in variable

        Returns
        -------
        None
        """
        session = get_session()
        if isinstance(parties, Party):
            parties = [parties]
        if not isinstance(suffix, tuple):
            suffix = (suffix,)
        tag = FederationTagNamespace.generate_tag(*suffix)

        for party in parties:
            if party.role not in self._dst:
                raise RuntimeError(
                    f"not allowed to remote object to {party} using {self._name}"
                )
        local = session.parties.local_party.role
        if local not in self._src:
            raise RuntimeError(
                f"not allowed to remote object from {local} using {self._name}"
            )

        name = self._short_name if self._use_short_name else self._name

        timer = profile.federation_remote_timer(name, self._name, tag, local, parties)
        session.federation.remote(
            v=obj, name=name, tag=tag, parties=parties, gc=self._remote_gc
        )
        timer.done(session.federation)

        self._remote_gc.gc()

    def get_parties(
        self,
        parties: Union[typing.List[Party], Party],
        suffix: Union[typing.Any, typing.Tuple] = tuple(),
    ):
        """
        get objects/tables from specified parties

        Parameters
        ----------
        parties: typing.List[Party]
           parties to remote object/table to
        suffix: str or tuple of str
           suffix used to distinguish federation with in variable

        Returns
        -------
        list
           a list of objects/tables get from parties with same order of ``parties``

        """
        session = get_session()
        if not isinstance(parties, list):
            parties = [parties]
        if not isinstance(suffix, tuple):
            suffix = (suffix,)
        tag = FederationTagNamespace.generate_tag(*suffix)

        for party in parties:
            if party.role not in self._src:
                raise RuntimeError(
                    f"not allowed to get object from {party} using {self._name}"
                )
        local = session.parties.local_party.role
        if local not in self._dst:
            raise RuntimeError(
                f"not allowed to get object to {local} using {self._name}"
            )

        name = self._short_name if self._use_short_name else self._name
        timer = profile.federation_get_timer(name, self._name, tag, local, parties)
        rtn = session.federation.get(
            name=name, tag=tag, parties=parties, gc=self._get_gc
        )
        timer.done(session.federation)

        self._get_gc.gc()

        return rtn

    def remote(self, obj, role=None, idx=-1, suffix=tuple()):
        """
        send obj to other parties.

        Args:
            obj: object to be sent
            role: role of parties to sent to, use one of ['Host', 'Guest', 'Arbiter', None].
                The default is None, means sent values to parties regardless their party role
            idx: id of party to sent to.
                The default is -1, which means sent values to parties regardless their party id
            suffix: additional tag suffix, the default is tuple()
        """
        party_info = get_parties()
        if idx >= 0 and role is None:
            raise ValueError("role cannot be None if idx specified")

        # get subset of dst roles in runtime conf
        if role is None:
            parties = party_info.roles_to_parties(self._dst, strict=False)
        else:
            if isinstance(role, str):
                role = [role]
            parties = party_info.roles_to_parties(role)

        if idx >= 0:
            if idx >= len(parties):
                raise RuntimeError(
                    f"try to remote to {idx}th party while only {len(parties)} configurated: {parties}, check {self._name}"
                )
            parties = parties[idx]
        return self.remote_parties(obj=obj, parties=parties, suffix=suffix)

    def get(self, idx=-1, role=None, suffix=tuple()):
        """
        get obj from other parties.

        Args:
            idx: id of party to get from.
                The default is -1, which means get values from parties regardless their party id
            suffix: additional tag suffix, the default is tuple()

        Returns:
            object or list of object
        """
        if role is None:
            src_parties = get_parties().roles_to_parties(
                roles=self._src, strict=False
            )
        else:
            if isinstance(role, str):
                role = [role]
            src_parties = get_parties().roles_to_parties(
                roles=role, strict=False
            )
        if isinstance(idx, list):
            rtn = self.get_parties(parties=[src_parties[i] for i in idx], suffix=suffix)
        elif isinstance(idx, int):
            if idx < 0:
                rtn = self.get_parties(parties=src_parties, suffix=suffix)
            else:
                if idx >= len(src_parties):
                    raise RuntimeError(
                        f"try to get from {idx}th party while only {len(src_parties)} configurated: {src_parties}, check {self._name}"
                    )
                rtn = self.get_parties(parties=src_parties[idx], suffix=suffix)[0]
        else:
            raise ValueError(
                f"illegal idx type: {type(idx)}, supported types: int or list of int"
            )
        return rtn


class BaseTransferVariables(object):
    def __init__(self, *args):
        pass

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    @staticmethod
    def set_flowid(flowid):
        """
        set global namespace for federations.

        Parameters
        ----------
        flowid: str
           namespace

        Returns
        -------
        None

        """
        FederationTagNamespace.set_namespace(str(flowid))

    def _create_variable(
        self, name: str, src: typing.Iterable[str], dst: typing.Iterable[str]
    ) -> Variable:
        full_name = f"{self.__module__}.{self.__class__.__name__}.{name}"
        return Variable.get_or_create(
            full_name, lambda: Variable(name=full_name, src=tuple(src), dst=tuple(dst))
        )

    @staticmethod
    def all_parties():
        """
        get all parties

        Returns
        -------
        list
           list of parties

        """
        return get_parties().all_parties

    @staticmethod
    def local_party():
        """
        indicate local party

        Returns
        -------
        Party
           party this program running on

        """
        return get_parties().local_party
