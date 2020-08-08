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

from arch.api.utils import file_utils, log_utils, core_utils
from fate_arch.common import Backend
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.entity.constant import StoreEngine
from arch.api.utils.conf_utils import get_base_config
import __main__


WORK_MODE = get_base_config('work_mode', 0)
BACKEND = Backend.EGGROLL
STORE_ENGINE=StoreEngine.EGGROLL
USE_LOCAL_DATABASE = get_base_config('use_local_database', True)

# upload data
USE_LOCAL_DATA = True

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

# FILE CONF
SERVER_CONF_PATH = 'conf/server_conf.json'

# job maximum number  of the initiator
MAX_CONCURRENT_JOB_RUN = 5
DEFAULT_TASK_PARALLELISM = 2
DEFAULT_PROCESSORS_PER_TASK = 10

# Limit the number of jobs on the host side
LIMIT_ROLE = 'host'
MAX_CONCURRENT_JOB_RUN_HOST = 5
RE_ENTRY_QUEUE_TIME = 2*60
RE_ENTRY_QUEUE_MAX = 60

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
DEFAULT_GRPC_OVERALL_TIMEOUT = 60 * 1000 * 60  # ms
JOB_DEFAULT_TIMEOUT = 7 * 24 * 60 * 60
DATABASE = get_base_config("database", {})
MODEL_STORE_ADDRESS = get_base_config("model_store_address", {})
HDFS_ADDRESS= ''
'''
Constants
'''
API_VERSION = "v1"
ROLE = 'fateflow'
SERVERS = 'servers'
MAIN_MODULE = os.path.relpath(__main__.__file__)
SERVER_MODULE = 'fate_flow_server.py'
TASK_EXECUTOR_MODULE = 'driver/task_executor.py'
TEMP_DIRECTORY = os.path.join(file_utils.get_project_base_directory(), "fate_flow", "temp")
HEADERS = {
    'Content-Type': 'application/json',
    'Connection': 'close'
}
DETECT_TABLE = ("fate_flow_detect_table_namespace", "fate_flow_detect_table_name", 16)

# endpoint
FATE_FLOW_MODEL_TRANSFER_ENDPOINT = '/v1/model/transfer'
FATE_MANAGER_GET_NODE_INFO_ENDPOINT = '/fate-manager/api/site/secretinfo'
FATE_MANAGER_NODE_CHECK_ENDPOINT = '/fate-manager/api/site/checksite'
FATE_BOARD_DASHBOARD_ENDPOINT = '/index.html#/dashboard?job_id={}&role={}&party_id={}'

# logger
log_utils.LoggerFactory.LEVEL = 10
# {CRITICAL: 50, FATAL:50, ERROR:40, WARNING:30, WARN:30, INFO:20, DEBUG:10, NOTSET:0}
log_utils.LoggerFactory.set_directory(os.path.join(file_utils.get_project_base_directory(), 'logs', 'fate_flow'))
stat_logger = log_utils.getLogger("fate_flow_stat")
detect_logger = log_utils.getLogger("fate_flow_detect")
access_logger = log_utils.getLogger("fate_flow_access")
data_manager_logger = log_utils.getLogger("fate_flow_data_manager")


"""
Services 
"""
IP = get_base_config("fate_flow", {}).get("host", "0.0.0.0")
HTTP_PORT = get_base_config("fate_flow", {}).get("http_port")
GRPC_PORT = get_base_config("fate_flow", {}).get("grpc_port")

# standalone job will be send to the standalone job server when FATE-Flow work on cluster deploy mode,
# but not the port for FATE-Flow on standalone deploy mode.
CLUSTER_STANDALONE_JOB_SERVER_PORT = 9381

# switch
ALIGN_TASK_INPUT_DATA_PARTITION_SWITCH = True

# init
RuntimeConfig.init_config(WORK_MODE=WORK_MODE)
RuntimeConfig.init_config(JOB_SERVER_HOST=core_utils.get_lan_ip(), HTTP_PORT=HTTP_PORT)
RuntimeConfig.init_config(BACKEND=BACKEND)
RuntimeConfig.init_config(STORE_ENGINE=STORE_ENGINE)
