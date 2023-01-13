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

import yaml
from pathlib import Path


__all__ = ["StatusCode", "FlowConfig", "StandaloneConfig", "SiteInfo"]


with Path(__file__).parent.parent.joinpath("pipeline_config.yaml").resolve().open("r") as fin:
    __DEFAULT_CONFIG: dict = yaml.safe_load(fin)


def get_default_config():
    return __DEFAULT_CONFIG


CONSOLE_DISPLAY_LOG = get_default_config().get("console_display_log", True)
if CONSOLE_DISPLAY_LOG is None:
    CONSOLE_DISPLAY_LOG = True


class StatusCode(object):
    SUCCESS = 0
    FAIL = 1
    CANCELED = 2


class SiteInfo(object):
    conf = get_default_config().get("site_info", {})
    ROLE = conf.get("local_role")
    PARTY_ID = conf.get("local_party_id")


class FlowConfig(object):
    conf = get_default_config().get("fate_flow", {})
    IP = conf.get("ip")
    PORT = conf.get("port")
    VERSION = conf.get("version")


class Device(object):
    def __init__(self, conf):
        self._type = conf.get("device", {}).get("type", "CPU")

    @property
    def type(self):
        return self._type


class LOGGER(object):
    def __init__(self, conf):
        self._level = conf.get("logger", {}).get("level", "DEBUG")
        self._debug_mode = conf.get("logger", {}).get("debug_mode", True)

    @property
    def level(self):
        return self._level

    @property
    def debug_mode(self):
        return self._debug_mode


class ComputingEngine(object):
    def __init__(self, conf):
        self._type = conf.get("computing_engine", {}).get("type", "standalone")

    @property
    def type(self):
        return self._type


class FederationEngine(object):
    def __init__(self, conf):
        self._type = conf.get("federation_engine", {}).get("type", "standalone")

    @property
    def type(self):
        return self._type


class MLMD(object):
    def __init__(self, conf):
        self._type = conf.get("mlmd", {}).get("type", "pipeline")
        self._db = conf.get("mlmd", {}).get("metadata", {}).get("db")
        if not self._db:
            default_path = Path.cwd()
            self._db = default_path.joinpath("pipeline_sqlite.db").as_uri()

    @property
    def db(self):
        return self._db

    @property
    def type(self):
        return self._type


class StandaloneConfig(object):
    conf = get_default_config().get("standalone", {})

    job_dir = conf.get("job_dir")
    if not job_dir:
        job_dir = Path.cwd()
    else:
        job_dir = Path(job_dir)

    JOB_CONF_DIR = job_dir.joinpath("jobs").as_uri()
    OUTPUT_DATA_DIR = job_dir.joinpath("data").as_uri()
    OUTPUT_MODEL_DIR = job_dir.joinpath("model").as_uri()
    OUTPUT_METRIC_DIR = job_dir.joinpath("metric").as_uri()
    OUTPUT_LOG_DIR = job_dir.joinpath("logs")

    MLMD = MLMD(conf)
    LOGGER = LOGGER(conf)
    DEVICE = Device(conf)
    COMPUTING_ENGINE = ComputingEngine(conf)
    FEDERATION_ENGINE = FederationEngine(conf)

