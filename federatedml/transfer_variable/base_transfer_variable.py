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

from arch.api import RuntimeInstance


class Variable(object):
    def __init__(self, name, transfer_variables):
        self.name = name
        self._transfer_variable = transfer_variables
        self._cleaner = None

    def generate_tag(self, *suffix):
        tag = self._transfer_variable.flowid
        if suffix:
            tag = f"{tag}.{'.'.join(map(str, suffix))}"
        return tag

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
        if not isinstance(suffix, tuple):
            suffix = (suffix,)

        obj = RuntimeInstance.TABLE_WRAPPER.unboxed(obj)
        if idx >= 0 and role is None:
            raise ValueError("role cannot be None if idx specified")
        if idx >= 0:
            dst_node = RuntimeInstance.FEDERATION.role_to_party(role=role, idx=idx)
            cleaner = RuntimeInstance.FEDERATION.remote(obj=obj,
                                                        name=self.name,
                                                        tag=self.generate_tag(*suffix),
                                                        parties=dst_node)
        else:
            if role is None:
                role = RuntimeInstance.FEDERATION.authorized_dst_roles(self.name)
            if isinstance(role, str):
                role = [role]
            dst_nodes = RuntimeInstance.FEDERATION.roles_to_parties(role)
            cleaner = RuntimeInstance.FEDERATION.remote(obj=obj,
                                                        name=self.name,
                                                        tag=self.generate_tag(*suffix),
                                                        parties=dst_nodes)

        if self._cleaner:
            self._cleaner.clean()
        self._cleaner = cleaner

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
        if not isinstance(suffix, tuple):
            suffix = (suffix,)
        tag = self.generate_tag(*suffix)
        name = self.name

        if self._cleaner:  # todo: fix bug, same tag
            self._cleaner.clean()

        src_role = RuntimeInstance.FEDERATION.authorized_src_roles(name)[0]
        if isinstance(idx, int):
            if idx < 0:
                src_parties = RuntimeInstance.FEDERATION.roles_to_parties(roles=[src_role])
                rtn, cleaner = RuntimeInstance.FEDERATION.get(name=name, tag=tag, parties=src_parties)
                rtn = [RuntimeInstance.TABLE_WRAPPER.boxed(value) for idx, value in enumerate(rtn)]
            else:
                src_node = RuntimeInstance.FEDERATION.role_to_party(role=src_role, idx=idx)
                rtn, cleaner = RuntimeInstance.FEDERATION.get(name=name, tag=tag, parties=src_node)
                rtn = RuntimeInstance.TABLE_WRAPPER.boxed(rtn[0])
        elif isinstance(idx, list):
            parties = [RuntimeInstance.FEDERATION.role_to_party(role=src_role, idx=pid) for pid in idx]
            rtn, cleaner = RuntimeInstance.FEDERATION.get(name=name, tag=tag, parties=parties)
            rtn = [RuntimeInstance.TABLE_WRAPPER.boxed(value) for idx, value in enumerate(rtn)]
        else:
            raise ValueError(f"illegal idx type: {type(idx)}, supported types: int or list of int")

        self._cleaner = cleaner
        return rtn


class BaseTransferVariables(object):
    def __init__(self, flowid=0):
        self.flowid = str(flowid)

    def set_flowid(self, flowid):
        self.flowid = flowid

    def _create_variable(self, name):
        return Variable(name=f"{self.__class__.__name__}.{name}", transfer_variables=self)
