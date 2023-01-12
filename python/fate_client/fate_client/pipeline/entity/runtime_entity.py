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
from .dag_structures import PartySpec


class Roles(object):
    def __init__(self):
        self._role_party_id_mappings = dict()
        self._role_party_index_mapping = dict()
        self._is_initialized = False

    def is_initialized(self):
        return self._is_initialized

    def set_role(self, role, party_id):
        if not isinstance(party_id, list):
            party_id = [party_id]

        if role not in self._role_party_id_mappings:
            self._role_party_id_mappings[role] = []
            self._role_party_index_mapping[role] = dict()

        for pid in party_id:
            if pid in self._role_party_index_mapping[role]:
                raise ValueError(f"role {role}, party {pid} has been added before")
            self._role_party_index_mapping[role][pid] = len(self._role_party_id_mappings[role])
            self._role_party_id_mappings[role].append(pid)

        self._role_party_id_mappings[role] = party_id
        self._is_initialized = True

    def get_party_id_list_by_role(self, role):
        return self._role_party_id_mappings[role]

    def get_party_by_role_index(self, role, index):
        return self._role_party_id_mappings[role][index]

    def get_runtime_roles(self):
        return self._role_party_id_mappings.keys()

    def get_parties_spec(self, roles=None):
        if not roles:
            roles = self._role_party_id_mappings.keys()

        roles = set(roles)

        role_list = []
        for role, party_id_list in self._role_party_id_mappings.items():
            if role not in roles:
                continue
            role_list.append(PartySpec(role=role, party_id=party_id_list))

        return role_list
