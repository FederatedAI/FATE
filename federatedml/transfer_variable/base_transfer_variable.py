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
from typing import Union

from arch.api import RuntimeInstance
from arch.api.transfer import Party


class Variable(object):
    def __init__(self, name, transfer_variables):
        self.name = name
        self._transfer_variable = transfer_variables
        self._get_cleaner = None
        self._remote_cleaner = None
        self._auto_clean = True

    def generate_tag(self, *suffix):
        tag = self._transfer_variable.flowid
        if suffix:
            tag = f"{tag}.{'.'.join(map(str, suffix))}"
        return tag

    def disable_auto_clean(self):
        self._auto_clean = False
        return self

    def clean(self):
        self._get_cleaner.maybe_clean()
        self._remote_cleaner.maybe_clean()

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
        tag = self.generate_tag(*suffix)
        cleaner = RuntimeInstance.FEDERATION.remote(obj=obj,
                                                    name=self.name,
                                                    tag=tag,
                                                    parties=parties)
        if self._remote_cleaner:
            if self._auto_clean:
                self._remote_cleaner.maybe_clean(tag)
                self._remote_cleaner = cleaner.update_tag(tag)
            else:
                self._remote_cleaner.extend(cleaner)
        else:
            self._remote_cleaner = cleaner.update_tag(tag)

    def get_parties(self, parties: Union[list, Party], suffix=tuple()):
        if not isinstance(suffix, tuple):
            suffix = (suffix,)
        tag = self.generate_tag(*suffix)

        if self._get_cleaner and self._auto_clean:
            self._get_cleaner.maybe_clean(tag)
        rtn, cleaner = RuntimeInstance.FEDERATION.get(name=self.name,
                                                      tag=self.generate_tag(*suffix),
                                                      parties=parties)
        rtn = [RuntimeInstance.TABLE_WRAPPER.boxed(value) for idx, value in enumerate(rtn)]

        if self._get_cleaner:
            if self._auto_clean:
                self._get_cleaner = cleaner.update_tag(tag)
            else:
                self._get_cleaner.extend(cleaner)
        else:
            self._get_cleaner = cleaner.update_tag(tag)
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
        src_role = self.authorized_src_roles[:1]
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
    def __init__(self, flowid=0):
        self.flowid = str(flowid)

    def set_flowid(self, flowid):
        self.flowid = flowid

    def _create_variable(self, name):
        return Variable(name=f"{self.__class__.__name__}.{name}", transfer_variables=self)

    @staticmethod
    def all_parties():
        return RuntimeInstance.FEDERATION.all_parties

    @staticmethod
    def local_party():
        return RuntimeInstance.FEDERATION.local_party
