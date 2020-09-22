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
# -*- coding: utf-8 -*-
import os

from fate_arch.computing import ComputingEngine
from fate_arch.federation import FederationEngine
from fate_arch.storage import StorageEngine
from fate_arch.common import file_utils, log, EngineType
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_arch.common.conf_utils import get_base_config
import __main__


WORK_MODE = get_base_config('work_mode', 0)
DATABASE = get_base_config("database", {})
MODEL_STORE_ADDRESS = get_base_config("model_store_address", {})

# storage engine is used for component output data
SUPPORT_ENGINES = {
    EngineType.COMPUTING: [ComputingEngine.EGGROLL, ComputingEngine.SPARK],
    EngineType.FEDERATION: [FederationEngine.EGGROLL, FederationEngine.RABBITMQ, FederationEngine.PROXY],
    EngineType.STORAGE: [StorageEngine.EGGROLL, StorageEngine.HDFS]
}

# upload data
UPLOAD_DATA_FROM_CLIENT = True

# Local authentication switch
USE_AUTHENTICATION = False
PRIVILEGE_COMMAND_WHITELIST = []

# Node check switch
CHECK_NODES_IDENTITY = False

# Registry
SERVICES_SUPPORT_REGISTRY = ["servings", "fateflow"]
FATE_SERVICES_REGISTERED_PATH = {
    "fateflow": "/FATE-SERVICES/flow/online/transfer/providers",
    "servings": "/FATE-SERVICES/serving/online/publishLoad/providers",
}

# default parameters
DEFAULT_TASK_PARALLELISM = 1
DEFAULT_TASK_CORES_PER_NODE = 5
DEFAULT_TASK_MEMORY_PER_NODE = 0  # mb
MAX_CORES_PERCENT_PER_JOB = 20  # 0 means not limited
STANDALONE_BACKEND_VIRTUAL_CORES_PER_NODE = 20
_ONE_DAY_IN_SECONDS = 60 * 60 * 24

# abnormal condition parameter
DEFAULT_GRPC_OVERALL_TIMEOUT = 2 * 60 * 1000  # ms
DEFAULT_FEDERATED_COMMAND_TRYS = 3
JOB_DEFAULT_TIMEOUT = 7 * 24 * 60 * 60
JOB_START_TIMEOUT = 60 * 1000  # ms

'''
Constants
'''
API_VERSION = "v1"
FATEFLOW_SERVICE_NAME = 'fateflow'
MAIN_MODULE = os.path.relpath(__main__.__file__)
SERVER_MODULE = 'fate_flow_server.py'
TEMP_DIRECTORY = os.path.join(file_utils.get_project_base_directory(), "temp", "fate_flow")
HEADERS = {
    'Content-Type': 'application/json',
    'Connection': 'close'
}

# endpoint
FATE_FLOW_MODEL_TRANSFER_ENDPOINT = '/v1/model/transfer'
FATE_MANAGER_GET_NODE_INFO_ENDPOINT = '/fate-manager/api/site/secretinfo'
FATE_MANAGER_NODE_CHECK_ENDPOINT = '/fate-manager/api/site/checksite'
FATE_BOARD_DASHBOARD_ENDPOINT = '/index.html#/dashboard?job_id={}&role={}&party_id={}'

# logger
log.LoggerFactory.LEVEL = 10
# {CRITICAL: 50, FATAL:50, ERROR:40, WARNING:30, WARN:30, INFO:20, DEBUG:10, NOTSET:0}
log.LoggerFactory.set_directory(os.path.join(file_utils.get_project_base_directory(), 'logs', 'fate_flow'))
stat_logger = log.getLogger("fate_flow_stat")
detect_logger = log.getLogger("fate_flow_detect")
access_logger = log.getLogger("fate_flow_access")
data_manager_logger = log.getLogger("fate_flow_data_manager")
peewee_logger = log.getLogger("peewee")


IP = get_base_config(FATEFLOW_SERVICE_NAME, {}).get("host", "127.0.0.1")
HTTP_PORT = get_base_config(FATEFLOW_SERVICE_NAME, {}).get("http_port")
GRPC_PORT = get_base_config(FATEFLOW_SERVICE_NAME, {}).get("grpc_port")

# switch
DEFAULT_FEDERATED_STATUS_COLLECT_TYPE = "PUSH"

# init
RuntimeConfig.init_config(WORK_MODE=WORK_MODE)
RuntimeConfig.init_config(JOB_SERVER_HOST=IP, HTTP_PORT=HTTP_PORT)
