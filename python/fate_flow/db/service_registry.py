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
from fate_arch.common import file_utils
from fate_arch.common.conf_utils import SERVICE_CONF
from fate_flow.settings import FATE_FLOW_SERVER_START_CONFIG_ITEMS, stat_logger
from .reload_config_base import ReloadConfigBase


class ServiceRegistry(ReloadConfigBase):
    FATEBOARD = None
    FATE_ON_STANDALONE = None
    FATE_ON_EGGROLL = None
    FATE_ON_SPARK = None
    MODEL_STORE_ADDRESS = None
    SERVINGS = None
    FATEMANAGER = None
    FATESTUDIO = None

    @classmethod
    def load(cls):
        path = Path(file_utils.get_project_base_directory()) / 'conf' / SERVICE_CONF
        conf = file_utils.load_yaml_conf(path)
        if not isinstance(conf, dict):
            raise ValueError('invalid config file')

        local_conf = {}
        local_path = path.with_name(f'local.{SERVICE_CONF}')
        if local_path.exists():
            local_conf = file_utils.load_yaml_conf(local_path)
            if not isinstance(local_conf, dict):
                raise ValueError('invalid local config file')

        cls.LINKIS_SPARK_CONFIG = conf.get('fate_on_spark', {}).get('linkis_spark')

        for k, v in conf.items():
            if k in FATE_FLOW_SERVER_START_CONFIG_ITEMS:
                stat_logger.info(f"{k} is fate flow server start config, pass load")
                pass
            else:
                if not isinstance(v, dict):
                    raise ValueError(f'{k} is not a dict, external services config must be a dict')
                setattr(cls, k.upper(), v)

        """
        for k, v in local_conf.items():
            if k == FATE_FLOW_SERVICE_NAME:
                if isinstance(v, dict):
                    for key, val in v.items():
                        key = key.upper()
                        if hasattr(cls, key):
                            setattr(cls, key, val)
            else:
                k = k.upper()
                if hasattr(cls, k) and type(getattr(cls, k)) == type(v):
                    setattr(cls, k, v)
        """
        return cls.get_all()

    @classmethod
    def register(cls, service_name, service_config):
        setattr(cls, service_name, service_config)

    @classmethod
    def write(cls):
        pass