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


# Server
API_VERSION = "v1"
FATEFLOW_SERVICE_NAME = "fateflow"
MAIN_MODULE = os.path.relpath(__main__.__file__)
SERVER_MODULE = "fate_flow_server.py"
TEMP_DIRECTORY = os.path.join(file_utils.get_project_base_directory(), "temp", "fate_flow")
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

# Registry
SERVICES_SUPPORT_REGISTRY = ["servings", "fateflow"]
FATE_SERVICES_REGISTERED_PATH = {
    "fateflow": "/FATE-SERVICES/flow/online/transfer/providers",
    "servings": "/FATE-SERVICES/serving/online/publishLoad/providers",
}

# Resource
TOTAL_CORES_OVERWEIGHT_PERCENT = 1  # 1 means no overweight
TOTAL_MEMORY_OVERWEIGHT_PERCENT = 1  # 1 means no overweight
DEFAULT_TASK_PARALLELISM = 1
DEFAULT_TASK_CORES = 4
DEFAULT_TASK_MEMORY = 0  # mb
MAX_CORES_PERCENT_PER_JOB = 1  # 1 means total
STANDALONE_BACKEND_VIRTUAL_CORES_PER_NODE = 20
IGNORE_RESOURCE_ROLES = {"arbiter"}
SUPPORT_IGNORE_RESOURCE_ENGINES = {ComputingEngine.EGGROLL, ComputingEngine.STANDALONE}

# Storage engine is used for component output data
SUPPORT_BACKENDS_ENTRANCE = {
    "fate_on_eggroll": {
        EngineType.COMPUTING: (ComputingEngine.EGGROLL, "clustermanager"),
        EngineType.STORAGE: (StorageEngine.EGGROLL, "clustermanager"),
        EngineType.FEDERATION: (FederationEngine.EGGROLL, "rollsite"),
    },
    "fate_on_spark_rabbitmq": {
        EngineType.COMPUTING: (ComputingEngine.SPARK, "spark"),
        EngineType.STORAGE: (StorageEngine.HDFS, "hdfs"),
        EngineType.FEDERATION: (FederationEngine.RABBITMQ, "rabbitmq"),
    },
    "fate_on_spark_pulsar": {
        EngineType.COMPUTING: (ComputingEngine.SPARK, "spark"),
        EngineType.STORAGE: (StorageEngine.HDFS, "hdfs"),
        EngineType.FEDERATION: (FederationEngine.PULSAR, "pulsar")
    }
}

# Scheduling
DEFAULT_REMOTE_REQUEST_TIMEOUT = 30 * 1000  # ms
DEFAULT_FEDERATED_COMMAND_TRYS = 3
JOB_DEFAULT_TIMEOUT = 3 * 24 * 60 * 60
JOB_START_TIMEOUT = 60 * 1000  # ms
END_STATUS_JOB_SCHEDULING_TIME_LIMIT = 5 * 60 * 1000 # ms
END_STATUS_JOB_SCHEDULING_UPDATES = 1

# Endpoint
FATE_FLOW_MODEL_TRANSFER_ENDPOINT = "/v1/model/transfer"
FATE_MANAGER_GET_NODE_INFO_ENDPOINT = "/fate-manager/api/site/secretinfo"
FATE_MANAGER_NODE_CHECK_ENDPOINT = "/fate-manager/api/site/checksite"
FATE_BOARD_DASHBOARD_ENDPOINT = "/index.html#/dashboard?job_id={}&role={}&party_id={}"

# Logger
log.LoggerFactory.LEVEL = 10
# {CRITICAL: 50, FATAL:50, ERROR:40, WARNING:30, WARN:30, INFO:20, DEBUG:10, NOTSET:0}
log.LoggerFactory.set_directory(os.path.join(file_utils.get_project_base_directory(), "logs", "fate_flow"))
stat_logger = log.getLogger("fate_flow_stat")
detect_logger = log.getLogger("fate_flow_detect")
access_logger = log.getLogger("fate_flow_access")
data_manager_logger = log.getLogger("fate_flow_data_manager")
peewee_logger = log.getLogger("peewee")

# Switch
UPLOAD_DATA_FROM_CLIENT = True
USE_AUTHENTICATION = False
PRIVILEGE_COMMAND_WHITELIST = []
CHECK_NODES_IDENTITY = False
DEFAULT_FEDERATED_STATUS_COLLECT_TYPE = get_base_config(FATEFLOW_SERVICE_NAME, {}).get("default_federated_status_collect_type", "PUSH")

# Init
RuntimeConfig.init_config(WORK_MODE=WORK_MODE)
RuntimeConfig.init_config(JOB_SERVER_HOST=IP, HTTP_PORT=HTTP_PORT)
