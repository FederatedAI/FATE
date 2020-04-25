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
import abc
from typing import Tuple, Union

from arch.api.base.utils.auth import FederationAuthorization
from arch.api.base.utils.clean import Rubbish
from arch.api.base.utils.consts import CONF_KEY_LOCAL, TRANSFER_CONF_PATH, ROLES
from arch.api.base.utils.party import Party


class Federation(object):

    def __init__(self, session_id, runtime_conf):
        if CONF_KEY_LOCAL not in runtime_conf:
            raise EnvironmentError("runtime_conf should be a dict containing key: {}".format(CONF_KEY_LOCAL))

        self._role = runtime_conf.get(CONF_KEY_LOCAL).get("role")
        self._party_id = runtime_conf.get(CONF_KEY_LOCAL).get("party_id")
        self._local_party = Party(self._role, self._party_id)
        self._session_id = session_id
        self._authorize = FederationAuthorization(TRANSFER_CONF_PATH)
        self._role_to_parties_map = {}
        self._all_parties = []
        for role in ROLES:
            party_id_list = runtime_conf.get('role').get(role, [])
            role_parties = [Party(role, party_id) for party_id in party_id_list]
            self._role_to_parties_map[role] = role_parties
            self._all_parties.extend(role_parties)

    @property
    def local_party(self):
        return self._local_party

    @property
    def all_parties(self):
        return self._all_parties

    def roles_to_parties(self, roles: list) -> list:
        nodes = []
        for role in roles:
            nodes.extend(self._role_to_parties_map[role])
        return nodes

    def role_to_party(self, role, idx) -> Party:
        return self._role_to_parties_map[role][idx]

    def authorized_src_roles(self, name) -> list:
        return self._authorize.authorized_src_roles(name)

    def authorized_dst_roles(self, name) -> list:
        return self._authorize.authorized_dst_roles(name)

    def _get_side_auth(self, name, parties):
        # check auth
        if self._role not in self._authorize.authorized_dst_roles(variable_name=name):
            raise PermissionError(f"try to get obj to {self._role}, with variable named {name}")
        roles = {party.role for party in parties}
        for role in roles:
            if role not in self._authorize.authorized_src_roles(variable_name=name):
                raise PermissionError(f"try to get obj from {role}, with variable named {name}")

    def _remote_side_auth(self, name, parties):
        # check auth
        if self._role not in self._authorize.authorized_src_roles(variable_name=name):
            raise PermissionError(f"try to remote obj from {self._role}, with variable named {name}")
        roles = {party.role for party in parties}
        for role in roles:
            if role not in self._authorize.authorized_dst_roles(variable_name=name):
                raise PermissionError(f"try to remote obj to {role}, with variable named {name}")

    @abc.abstractmethod
    def remote(self, obj, name: str, tag: str, parties: Union[Party, list]) -> Rubbish:
        """
        remote object or dtable to parties identified by `parties`,
        through transfer_variable identified by name and tag.

        :param obj: dtable or object to be remote
        :param name: name of transfer_variable
        :param tag: tag of transfer_variable
        :param parties: instance or instances of `Party`, specify the parties to send obj to
        :return: an instance of Cleaner for latter clean tasks.
        """
        pass

    @abc.abstractmethod
    def get(self, name: str, tag: str, parties: Union[Party, list]) -> Tuple[list, Rubbish]:
        """
         get object or dtable from parties identified by `parties`,
         through transfer_variable identified by name and tag.
        :param name: name of transfer_variable
        :param tag: tag of transfer_variable
        :param parties: instance of Party or list of instances of parties, specify the parties to get obj from
        :return: a tuple, with a list of results and an instance of Cleaner for latter clean tasks.
        """
        pass
