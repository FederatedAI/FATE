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
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path

from ruamel import yaml

temperate = """\
# 0 for standalone, 1 for cluster
work_mode: 0
# 0 for eggroll, 1 for spark
backend: 0
# base dir for data upload conf eg
# examples/data/breast_hetero_guest.csv -> $data_base_dir/examples/data/breast_hetero_guest.csv
data_base_dir: ../../../
parties:
  guest: [10000]
  host: [9999, 10000]
  arbiter: [9999]
services:
  - flow_services:
      - {address: 127.0.0.1:9380, parties: [9999, 10000]}
    ssh_tunnel: # optional
      enable: false
      ssh_address: <remote ip>:<remote port>
      ssh_username:
      ssh_password: # optional
      ssh_priv_key: "~/.ssh/id_rsa"

# what is ssh_tunnel?
# to open the ssh tunnel(s) if the remote service
# cannot be accessed directly from the location where the test suite is run!
#
#                       +---------------------+
#                       |    ssh address      |
#                       |    ssh username     |
#                       |    ssh password/    |
#         +--------+    |    ssh priv_key     |        +----------------+
#         |local ip+----------ssh tuunel-------------->+remote local ip |
#         +--------+    |                     |        +----------------+
#                       |                     |
# request local ip:port +----- as if --------->request remote's local ip:port from remote side
#                       |                     |
#                       |                     |
#                       +---------------------+
#

"""

default_config = Path(__file__).parent.joinpath("fate_test_config.yaml").resolve()


def create_config(path: Path, override=False):
    if path.exists() and not override:
        raise FileExistsError(f"{path} exists")
    with path.open("w") as f:
        f.write(temperate)


def priority_config():
    if Path("fate_test_config.yaml").exists():
        return Path("fate_test_config.yaml").resolve()
    if not default_config.exists():
        create_config(default_config)
    return default_config


@dataclass
class Parties(object):
    guest: typing.List[int]
    host: typing.List[int]
    arbiter: typing.List[int] = field(default_factory=list)

    @staticmethod
    def from_dict(d: dict):
        return Parties(**d)

    def __post_init__(self):
        self._party_to_role_string = {}
        for role in self.__class__.__annotations__:
            parties = getattr(self, role)
            for i, party in enumerate(parties):
                if party not in self._party_to_role_string:
                    self._party_to_role_string[party] = set()
                self._party_to_role_string[party].add(f"{role.lower()}_{i}")

    def party_to_role_string(self, party):
        return self._party_to_role_string[party]

    def extract_role(self, counts: typing.MutableMapping[str, int]):
        roles = {}
        for role, num in counts.items():
            if not hasattr(self, role):
                raise ValueError(f"{role} should be one of [guest, host, arbiter]")
            else:
                if len(getattr(self, role)) < num:
                    raise ValueError(f"require {num} {role} parties, only {len(getattr(self, role))} in config")
            roles[role] = getattr(self, role)[:num]
        return roles

    def extract_initiator_role(self, role):
        initiator_role = role.strip()
        if len(getattr(self, initiator_role)) < 1:
            raise ValueError(f"role {initiator_role} has empty party list")
        party_id = getattr(self, initiator_role)[0]
        return dict(role=initiator_role, party_id=party_id)


@dataclass
class Config(object):
    service = namedtuple("service", ["address"])
    tunnel_service = namedtuple("tunnel_service", ["tunnel_id", "index"])
    tunnel = namedtuple("tunnel", ["ssh_address", "ssh_username", "ssh_password", "ssh_priv_key", "services_address"])

    def __init__(self, config):
        self.work_mode = config["work_mode"]
        self.backend = config["backend"]
        self.data_base_dir = config["data_base_dir"]
        self.parties = Parties.from_dict(config["parties"])
        self.party_to_service_id = {}
        self.service_id_to_service = {}
        self.tunnel_id_to_tunnel = {}

        tunnel_id = 0
        service_id = 0
        for service_config in config["services"]:
            flow_services = service_config["flow_services"]
            if service_config.get("ssh_tunnel", {}).get("enable", False):
                tunnel_id += 1
                services_address = []
                for index, flow_service in enumerate(flow_services):
                    service_id += 1
                    address_host, address_port = flow_service["address"].split(":")
                    address_port = int(address_port)
                    services_address.append((address_host, address_port))
                    self.service_id_to_service[service_id] = self.tunnel_service(tunnel_id, index)
                    for party in flow_service["parties"]:
                        self.party_to_service_id[party] = service_id
                tunnel_config = service_config["ssh_tunnel"]
                ssh_address_host, ssh_address_port = tunnel_config["ssh_address"].split(":")
                self.tunnel_id_to_tunnel[tunnel_id] = self.tunnel((ssh_address_host, int(ssh_address_port)),
                                                                  tunnel_config["ssh_username"],
                                                                  tunnel_config["ssh_password"],
                                                                  tunnel_config["ssh_priv_key"],
                                                                  services_address)
            else:
                for flow_service in flow_services:
                    service_id += 1
                    address = flow_service["address"]
                    self.service_id_to_service[service_id] = self.service(address)
                    for party in flow_service["parties"]:
                        self.party_to_service_id[party] = service_id

    @staticmethod
    def load(path: typing.Union[str, Path], **kwargs):
        if isinstance(path, str):
            path = Path(path)
        config = {}
        if path is not None:
            with path.open("r") as f:
                config.update(yaml.safe_load(f))
        config["data_base_dir"] = path.resolve().joinpath(config["data_base_dir"]).resolve()
        config.update(kwargs)
        return Config(config)
