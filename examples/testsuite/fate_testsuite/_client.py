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

from pathlib import Path

import sshtunnel

from fate_testsuite._flow_client import FLOWClient
from fate_testsuite._io import LOGGER
from fate_testsuite._parser import Config


class Clients(object):
    def __init__(self, config: Config):
        self.config = config
        self._flow_clients = {}

        # create flow clients for local
        for local_service in self.config.local_services:
            flow_client = FLOWClient(local_service.address.string_format, self.config.data_base_dir)
            for role_str in self.config.parties.parties_to_role_string(local_service.parties):
                self._flow_clients[role_str] = flow_client

    def __getitem__(self, role_str: str) -> 'FLOWClient':
        if role_str not in self._flow_clients:
            raise RuntimeError(f"no flow client found binding to {role_str}")
        return self._flow_clients[role_str]

    def __enter__(self):
        # open ssh tunnels and create flow clients for remote
        self._tunnels = []
        for tunnel_conf in self.config.ssh_tunnel:
            role_strings = []
            remote_bind_addresses = []
            for service in tunnel_conf.services:
                role_strings.append(self.config.parties.parties_to_role_string(service.parties))
                remote_bind_addresses.append(service.address.tuple_format)

            tunnel = sshtunnel.SSHTunnelForwarder(ssh_address_or_host=tunnel_conf.ssh_address.tuple_format,
                                                  ssh_username=tunnel_conf.ssh_username,
                                                  ssh_password=tunnel_conf.ssh_password,
                                                  ssh_pkey=tunnel_conf.ssh_priv_key,
                                                  remote_bind_addresses=remote_bind_addresses)
            tunnel.start()
            for role_strings, address in zip(role_strings, tunnel.local_bind_addresses):
                client = FLOWClient(address=f"127.0.0.1:{address[1]}", data_base_dir=self.config.data_base_dir)
                for role_string in role_strings:
                    self._flow_clients[role_string] = client
            self._tunnels.append(tunnel)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for tunnel in self._tunnels:
            try:
                tunnel.stop()
            except Exception as e:
                LOGGER.exception(e)
