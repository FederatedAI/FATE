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
import typing
from typing import Union

from arch.api import RuntimeInstance
from arch.api.base.utils.clean import Cleaner
from arch.api.base.utils.party import Party


class TransferNameSpace(object):
    __namespace = "default"

    @classmethod
    def set_namespace(cls, namespace):
        cls.__namespace = namespace

    @classmethod
    def generate_tag(cls, *suffix):
        tags = (cls.__namespace, *map(str, suffix))
        return '.'.join(tags)


class Variable(object):
    def __init__(self, name: str,
                 src: typing.Tuple[str],
                 dst: typing.Tuple[str]):
        self.name = name
        self._src = src
        self._dst = dst
        self._get_cleaner = Cleaner()
        self._remote_cleaner = Cleaner()
        self._auto_clean = True
        self._preserve_num = 2

    def set_preserve_num(self, n):
        self._preserve_num = n
        return self

    def get_preserve_num(self):
        return self._preserve_num

    def disable_auto_clean(self):
        self._auto_clean = False
        return self

    def clean(self):
        self._get_cleaner.clean_all()
        self._remote_cleaner.clean_all()

    @staticmethod
    def roles_to_parties(roles):
        return RuntimeInstance.FEDERATION.roles_to_parties(roles=roles)

    @property
    def authorized_dst_roles(self):
        return RuntimeInstance.FEDERATION.authorized_dst_roles(self.name)

    @property
    def authorized_src_roles(self):
        return RuntimeInstance.FEDERATION.authorized_src_roles(self.name)

    def remote_parties(self, obj, parties: Union[list, Party], suffix=tuple()):
        if not isinstance(suffix, tuple):
            suffix = (suffix,)
        obj = RuntimeInstance.TABLE_WRAPPER.unboxed(obj)
        tag = TransferNameSpace.generate_tag(*suffix)
        rubbish = RuntimeInstance.FEDERATION.remote(obj=obj,
                                                    name=self.name,
                                                    tag=tag,
                                                    parties=parties)
        self._remote_cleaner.push(rubbish)
        if self._auto_clean:
            self._remote_cleaner.keep_latest_n(self._preserve_num)

    def get_parties(self, parties: Union[list, Party], suffix=tuple()):
        if not isinstance(suffix, tuple):
            suffix = (suffix,)
        tag = TransferNameSpace.generate_tag(*suffix)

        if self._auto_clean:
            if self._get_cleaner.is_latest_tag(tag):
                self._get_cleaner.keep_latest_n(self._preserve_num)
            else:
                self._get_cleaner.keep_latest_n(self._preserve_num - 1)

        rtn, rubbish = RuntimeInstance.FEDERATION.get(name=self.name,
                                                      tag=TransferNameSpace.generate_tag(*suffix),
                                                      parties=parties)
        rtn = [RuntimeInstance.TABLE_WRAPPER.boxed(value) for idx, value in enumerate(rtn)]

        self._get_cleaner.push(rubbish)
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
        if idx >= 0 and role is None:
            raise ValueError("role cannot be None if idx specified")
        if role is None:
            role = self.authorized_dst_roles
        if isinstance(role, str):
            role = [role]
        parties = self.roles_to_parties(role)
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
        src_role = self.authorized_src_roles
        src_parties = self.roles_to_parties(roles=src_role)
        if isinstance(idx, list):
            rtn = self.get_parties(parties=[src_parties[i] for i in idx], suffix=suffix)
        elif isinstance(idx, int):
            rtn = self.get_parties(parties=src_parties, suffix=suffix) if idx < 0 else \
                self.get_parties(parties=src_parties[idx], suffix=suffix)[0]
        else:
            raise ValueError(f"illegal idx type: {type(idx)}, supported types: int or list of int")
        return rtn


class BaseTransferVariables(object):
    __singleton = True
    __instance = {}

    def __init__(self, *args):
        pass

    @classmethod
    def _disable__singleton(cls):
        cls.__singleton = False

    def __new__(cls, *args, **kwargs):
        if cls.__singleton:
            if cls.__name__ not in cls.__instance:
                cls.__instance[cls.__name__] = object.__new__(cls)
            return cls.__instance[cls.__name__]
        else:
            return object.__new__(cls)

    # noinspection PyMethodMayBeStatic
    def set_flowid(self, flowid):
        TransferNameSpace.set_namespace(str(flowid))

    def _create_variable(self, name: str, src: typing.Iterable[str], dst: typing.Iterable[str]):
        return Variable(name=f"{self.__class__.__name__}.{name}", src=tuple(src), dst=tuple(dst))

    @staticmethod
    def all_parties():
        return RuntimeInstance.FEDERATION.all_parties

    @staticmethod
    def local_party():
        return RuntimeInstance.FEDERATION.local_party
