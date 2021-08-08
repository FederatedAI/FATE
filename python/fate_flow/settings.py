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
import os
from pathlib import Path

from fate_arch.common import file_utils, log
from fate_arch.common.conf_utils import get_base_config
from fate_arch.computing import ComputingEngine
from fate_arch.computing import ComputingEngine
from fate_arch.federation import FederationEngine
from fate_arch.storage import StorageEngine
from fate_arch.common import file_utils, log, EngineType
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_arch.common.conf_utils import SERVICE_CONF


# Server
API_VERSION = "v1"
FATEFLOW_SERVICE_NAME = "fateflow"
SERVER_MODULE = "fate_flow_server.py"
TEMP_DIRECTORY = os.path.join(
    file_utils.get_project_base_directory(), "temp", "fate_flow")
FATE_FLOW_DIRECTORY = os.path.join(file_utils.get_python_base_directory(), "fate_flow")
SUBPROCESS_STD_LOG_NAME = "std.log"
HEADERS = {
    "Content-Type": "application/json",
    "Connection": "close",
    "service": FATEFLOW_SERVICE_NAME
}
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
GRPC_SERVER_MAX_WORKERS = None

IP = get_base_config(FATEFLOW_SERVICE_NAME, {}).get("host", "127.0.0.1")
HTTP_PORT = get_base_config(FATEFLOW_SERVICE_NAME, {}).get("http_port")
GRPC_PORT = get_base_config(FATEFLOW_SERVICE_NAME, {}).get("grpc_port")

WORK_MODE = get_base_config("work_mode", 0)
DATABASE = get_base_config("database", {})
MODEL_STORE_ADDRESS = get_base_config("model_store_address", {})

MAX_TIMESTAMP_INTERVAL = 60

# Registry
FATE_SERVICES_REGISTRY = {
    'zookeeper': {
        'fateflow': "/FATE-SERVICES/flow/online/transfer/providers",
        'servings': "/FATE-SERVICES/serving/online/publishLoad/providers",
    },
}

# Engine
IGNORE_RESOURCE_COMPUTING_ENGINE = {
    ComputingEngine.LINKIS_SPARK
}

IGNORE_RESOURCE_ROLES = {"arbiter"}

SUPPORT_IGNORE_RESOURCE_ENGINES = {
    ComputingEngine.EGGROLL, ComputingEngine.STANDALONE
}

# linkis spark config
LINKIS_SPARK_CONFIG = get_base_config("fate_on_spark", {}).get("linkis_spark")
LINKIS_EXECUTE_ENTRANCE = "/api/rest_j/v1/entrance/execute"
LINKIS_KILL_ENTRANCE = "/api/rest_j/v1/entrance/execID/kill"
LINKIS_QUERT_STATUS = "/api/rest_j/v1/entrance/execID/status"
LINKIS_SUBMIT_PARAMS = {
    "configuration": {
        "startup": {
            "spark.python.version": "/data/anaconda3/bin/python",
            "archives": "hdfs:///apps-data/fate/python.zip#python,hdfs:///apps-data/fate/fate_guest.zip#fate_guest",
            "spark.executorEnv.PYTHONPATH": "./fate_guest/python:$PYTHONPATH",
            "wds.linkis.rm.yarnqueue": "dws",
            "spark.pyspark.python": "python/bin/python"
        }
    }
}
LINKIS_RUNTYPE = "py"
LINKIS_LABELS = {"tenant": "fate"}

# Endpoint
FATE_FLOW_MODEL_TRANSFER_ENDPOINT = "/v1/model/transfer"
FATE_MANAGER_GET_NODE_INFO_ENDPOINT = "/fate-manager/api/site/secretinfo"
FATE_MANAGER_NODE_CHECK_ENDPOINT = "/fate-manager/api/site/checksite"
FATE_BOARD_DASHBOARD_ENDPOINT = "/index.html#/dashboard?job_id={}&role={}&party_id={}"

# Logger
log.LoggerFactory.LEVEL = 10
# {CRITICAL: 50, FATAL:50, ERROR:40, WARNING:30, WARN:30, INFO:20, DEBUG:10, NOTSET:0}
log.LoggerFactory.set_directory(os.path.join(
    file_utils.get_project_base_directory(), "logs", "fate_flow"))
stat_logger = log.getLogger("fate_flow_stat")
detect_logger = log.getLogger("fate_flow_detect")
access_logger = log.getLogger("fate_flow_access")
data_manager_logger = log.getLogger("fate_flow_data_manager")
peewee_logger = log.getLogger("peewee")

# Switch
# upload
UPLOAD_DATA_FROM_CLIENT = True

# authentication
USE_AUTHENTICATION = False
USE_DATA_AUTHENTICATION = False
AUTOMATIC_AUTHORIZATION_OUTPUT_DATA = True
USE_DEFAULT_TIMEOUT = False
AUTHENTICATION_DEFAULT_TIMEOUT = 30 * 24 * 60 * 60 # s
PRIVILEGE_COMMAND_WHITELIST = []

CHECK_NODES_IDENTITY = False


class Settings:

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

        cls.IP = conf.get(FATEFLOW_SERVICE_NAME, {}).pop('host', '127.0.0.1')
        cls.LINKIS_SPARK_CONFIG = conf.get('fate_on_spark', {}).get('linkis_spark')

        for k, v in conf.items():
            if k == FATEFLOW_SERVICE_NAME:
                if not isinstance(v, dict):
                    raise ValueError(f'{FATEFLOW_SERVICE_NAME} is not a dict')

                for key, val in v.items():
                    setattr(cls, key.upper(), val)
            else:
                setattr(cls, k.upper(), v)

        for k, v in local_conf.items():
            if k == FATEFLOW_SERVICE_NAME:
                if isinstance(v, dict):
                    for key, val in v.items():
                        key = key.upper()
                        if hasattr(cls, key):
                            setattr(cls, key, val)
            else:
                k = k.upper()
                if hasattr(cls, k) and type(getattr(cls, k)) == type(v):
                    setattr(cls, k, v)

        RuntimeConfig.init_config(WORK_MODE=cls.WORK_MODE, JOB_SERVER_HOST=cls.IP, HTTP_PORT=cls.HTTP_PORT)


Settings.load()

# 1.7 merge
DEFAULT_FEDERATED_STATUS_COLLECT_TYPE = get_base_config(
    FATEFLOW_SERVICE_NAME, {}).get("default_federated_status_collect_type", "PUSH")