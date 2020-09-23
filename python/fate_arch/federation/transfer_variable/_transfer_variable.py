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
from typing import Union

from fate_arch.common import Party, profile
from fate_arch.common.log import getLogger
from fate_arch.federation.transfer_variable._auth import _check_variable_auth_conf
from fate_arch.federation.transfer_variable._cleaner import IterationGC
from fate_arch.federation.transfer_variable._namespace import FederationTagNamespace
from fate_arch.session import get_latest_opened

LOGGER = getLogger()


class Variable(object):
    __disable_auth_check = False
    __instances: typing.MutableMapping[str, 'Variable'] = {}

    @classmethod
    def _disable_auth_check(cls):
        """
        used in auth conf generation, don't call this in real application
        """
        cls.__disable_auth_check = True

    @classmethod
    def get_or_create(cls, name, create_func: typing.Callable[[], 'Variable']) -> 'Variable':
        if name not in cls.__instances:
            value = create_func()
            cls.__instances[name] = value
        return cls.__instances[name]

    def __init__(self, name: str,
                 src: typing.Tuple[str, ...],
                 dst: typing.Tuple[str, ...]):

        if name in self.__instances:
            raise RuntimeError(
                f"{self.__instances[name]} with {name} already initialized, which expected to be an singleton object.")

        if not self.__disable_auth_check:
            auth_src, auth_dst = _check_variable_auth_conf(name)
            if set(src) != set(auth_src) or set(dst) != set(auth_dst):
                raise RuntimeError(f"Variable {name} auth error, "
                                   f"acquired: src={src}, dst={dst}, allowed: src={auth_src}, dst={auth_dst}")

        assert len(name.split(".")) >= 3, "incorrect name format, should be `module_name.class_name.variable_name`"
        self._name = name
        self._src = src
        self._dst = dst
        self._get_gc = IterationGC()
        self._remote_gc = IterationGC()
        self._use_short_name = True
        self._short_name = self._get_short_name(self._name)

    @staticmethod
    def _get_short_name(name):
        fix_sized = hashlib.blake2b(name.encode('utf-8'), digest_size=10).hexdigest()
        _, right = name.split('.', 1)
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

    def remote_parties(self,
                       obj,
                       parties: Union[typing.List[Party], Party],
                       suffix: Union[typing.Any, typing.Tuple] = tuple()):
        session = get_latest_opened()
        if isinstance(parties, Party):
            parties = [parties]
        if not isinstance(suffix, tuple):
            suffix = (suffix,)
        tag = FederationTagNamespace.generate_tag(*suffix)

        for party in parties:
            if party.role not in self._dst:
                raise RuntimeError(f"not allowed to remote object to {party} using {self._name}")
        local = session.parties.local_party.role
        if local not in self._src:
            raise RuntimeError(f"not allowed to remote object from {local} using {self._name}")

        name = self._short_name if self._use_short_name else self._name

        timer = profile.federation_remote_timer(name, self._name, tag, local, parties)
        session.federation.remote(v=obj, name=name, tag=tag, parties=parties, gc=self._remote_gc)
        timer.done(session.federation)

        self._remote_gc.gc()

    def get_parties(self,
                    parties: Union[typing.List[Party], Party],
                    suffix: Union[typing.Any, typing.Tuple] = tuple()):
        session = get_latest_opened()
        if not isinstance(parties, list):
            parties = [parties]
        if not isinstance(suffix, tuple):
            suffix = (suffix,)
        tag = FederationTagNamespace.generate_tag(*suffix)

        for party in parties:
            if party.role not in self._src:
                raise RuntimeError(f"not allowed to get object from {party} using {self._name}")
        local = session.parties.local_party.role
        if local not in self._dst:
            raise RuntimeError(f"not allowed to get object to {local} using {self._name}")

        name = self._short_name if self._use_short_name else self._name
        timer = profile.federation_get_timer(name, self._name, tag, local, parties)
        rtn = session.federation.get(name=name, tag=tag, parties=parties, gc=self._get_gc)
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
        party_info = get_latest_opened().parties
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
            parties = parties[idx]
        return self.remote_parties(obj=obj, parties=parties, suffix=suffix)

    def get(self, idx=-1, suffix=tuple()):
        """
        get obj from other parties.

        Args:
            idx: id of party to get from.
                The default is -1, which means get values from parties regardless their party id
            suffix: additional tag suffix, the default is tuple()

        Returns:
            object or list of object
        """
        src_parties = get_latest_opened().parties.roles_to_parties(roles=self._src, strict=False)
        if isinstance(idx, list):
            rtn = self.get_parties(parties=[src_parties[i] for i in idx], suffix=suffix)
        elif isinstance(idx, int):
            rtn = self.get_parties(parties=src_parties, suffix=suffix) if idx < 0 else \
                self.get_parties(parties=src_parties[idx], suffix=suffix)[0]
        else:
            raise ValueError(f"illegal idx type: {type(idx)}, supported types: int or list of int")
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
        FederationTagNamespace.set_namespace(str(flowid))

    def _create_variable(self, name: str, src: typing.Iterable[str], dst: typing.Iterable[str]) -> Variable:
        full_name = f"{self.__module__}.{self.__class__.__name__}.{name}"
        return Variable.get_or_create(full_name, lambda: Variable(name=full_name, src=tuple(src), dst=tuple(dst)))

    @staticmethod
    def all_parties():
        return get_latest_opened().parties.all_parties

    @staticmethod
    def local_party():
        return get_latest_opened().parties.local_party
