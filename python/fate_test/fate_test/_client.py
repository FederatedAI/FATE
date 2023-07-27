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

from fate_test._flow_client import FLOWClient
from fate_test._parser import Config


class Clients(object):
    def __init__(self, config: Config):
        self._flow_clients = {}
        # self._tunnel_id_to_flow_clients = {}
        self._role_str_to_service_id = {}
        # self._service_id_to_role_str = {}
        # self._service_id_to_party = {}
        # self._tunnel_id_to_tunnel = config.tunnel_id_to_tunnel
        for party, service_id in config.party_to_service_id.items():
            for role_str in config.parties.party_to_role_string(party):
                self._role_str_to_service_id[role_str] = service_id
                # self._service_id_to_role_str[service_id] = role_str
                # self._service_id_to_party[service_id] = party

        for service_id, service in config.service_id_to_service.items():
            if isinstance(service, Config.service):
                self._flow_clients[service_id] = FLOWClient(
                    service.address, config.data_base_dir, config.cache_directory)

            """elif isinstance(service, Config.tunnel_service):
                self._flow_clients[service_id] = FLOWClient(None, config.data_base_dir, config.cache_directory)
                self._tunnel_id_to_flow_clients.setdefault(service.tunnel_id, []).append(
                    (service.index, self._flow_clients[service_id]))"""

    def __getitem__(self, role_str: str) -> 'FLOWClient':
        if role_str not in self._role_str_to_service_id:
            raise RuntimeError(f"no flow client found binding to {role_str}")
        return self._flow_clients[self._role_str_to_service_id[role_str]]

    def contains(self, role_str):
        return role_str in self._role_str_to_service_id

    def all_roles(self):
        return sorted(self._role_str_to_service_id.keys())
