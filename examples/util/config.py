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

from ruamel import yaml


class Parties(object):
    def __init__(self, parties):
        if len(parties) == 0:
            raise ValueError(f"Parties id must be specified.")
        self.host = parties.get("host", None)
        self.guest = parties.get("guest", None)
        self.arbiter = parties.get("arbiter", None)


class Config(object):
    def __init__(self, config):
        self.parties = Parties(config.get("parties", {}))
        self.backend = config.get("backend", 0)
        self.work_mode = config.get("work_mode", 0)

    @staticmethod
    def load(path):
        if isinstance(path, str):
            path = Path(path)
        conf = {}
        if path is not None:
            with open(path, "r") as f:
                conf.update(yaml.safe_load(f))
        return Config(conf)
