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

temperate = """\
# ssh tunnel forwarder, request remote services through local port
# uncomment flowing and modify fields according to there comments if at least one
# FATE party is unreachable from where you running testsuite.
#ssh_tunnel:
#  - ssh_address:  172.0.0.1:22 # format: ip:port, ssh address to login,
#    ssh_username: app  # ssh username
#    ssh_password: ??  # ssh password, optional, if `ssh_priv_key` provided
#    ssh_priv_key: "~/.ssh/id_rsa"  # ssh priv key path, optional, if `ssh_password` provided
#    services:  # flow services to bind
#      - address: 127.0.0.1:9380
#        parties: [9999]

# flow services in local, could be comments out
# if all `FATE` server are unreachable from  where you running testsuite.
local_services:
  - address: 127.0.0.1:9380
    parties: [9999, 10000]

# assignment parties
parties:
  guest:
    - 10000
  host:
    - 9999
    - 10000
  arbiter:
    - 9999

# work mode, accept 0 or 1, dependents on `FATE` deployed mode.
work_mode: 0

# base dir of data file path in data conf
# we need these path to correct the data file path for testsuite.
# eg. for flowing data config:
#  {
#    "file": "examples/data/breast_hetero_guest.csv",
#    "head": 1,
#    "partition": 16,
#    "work_mode": 0,
#    "table_name": "breast_hetero_guest",
#    "namespace": "experiment",
#  }
# we need to reinterpret as `$data_base_dir/examples/data/breast_hetero_guest.csv`
# dir could be relative to these config.
data_base_dir: ../../../

"""

default_config = Path(__file__).parent.joinpath("testsuite_config.yaml").resolve()


def create_config(path: Path, override=False):
    if path.exists() and not override:
        raise FileExistsError(f"{path} exists")
    with path.open("w") as f:
        f.write(temperate)


def priority_config():
    if Path("testsuite_config.yaml").exists():
        return Path("testsuite_config.yaml").resolve()
    if not default_config.exists():
        create_config(default_config)
    return default_config
