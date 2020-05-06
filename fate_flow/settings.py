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

from arch.api import Backend
from arch.api.utils import file_utils, log_utils, core_utils
from fate_flow.entity.runtime_config import RuntimeConfig
from arch.api.utils.core_utils import get_lan_ip
from arch.api.utils.conf_utils import get_base_config
import __main__

from fate_flow.utils.setting_utils import CenterConfig


WORK_MODE = get_base_config('work_mode', 0)
BACKEND = Backend.EGGROLL
USE_LOCAL_DATABASE = True

# upload data
USE_LOCAL_DATA = True

# Local authentication switch
USE_AUTHENTICATION = False
PRIVILEGE_COMMAND_WHITELIST = []

# Node check switch
CHECK_NODES_IDENTITY = False

# zookeeper
USE_CONFIGURATION_CENTER = False
ZOOKEEPER_HOSTS = ['127.0.0.1:2181']

MAX_CONCURRENT_JOB_RUN = 5
MAX_CONCURRENT_JOB_RUN_HOST = 5
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
DEFAULT_GRPC_OVERALL_TIMEOUT = 60 * 1000 * 60  # ms
JOB_DEFAULT_TIMEOUT = 7 * 24 * 60 * 60
DATABASE = get_base_config("database", {})
DEFAULT_MODEL_STORE_ADDRESS = get_base_config("default_model_store_address", {})

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
# fate-serving
SERVINGS_ZK_PATH = '/FATE-SERVICES/serving/online/publishLoad/providers'
FATE_FLOW_ZK_PATH = '/FATE-SERVICES/flow/online/transfer/providers'
FATE_FLOW_MODEL_TRANSFER_PATH = '/v1/model/transfer'
# fate-manager
FATE_MANAGER_GET_NODE_INFO = '/node/info'
FATE_MANAGER_NODE_CHECK = '/node/management/check'

# logger
log_utils.LoggerFactory.LEVEL = 10
# {CRITICAL: 50, FATAL:50, ERROR:40, WARNING:30, WARN:30, INFO:20, DEBUG:10, NOTSET:0}
log_utils.LoggerFactory.set_directory(os.path.join(file_utils.get_project_base_directory(), 'logs', 'fate_flow'))
stat_logger = log_utils.getLogger("fate_flow_stat")
detect_logger = log_utils.getLogger("fate_flow_detect")
access_logger = log_utils.getLogger("fate_flow_access")
audit_logger = log_utils.audit_logger()

"""
Services 
"""
IP = get_base_config("fate_flow", {}).get("host", "0.0.0.0")
HTTP_PORT = get_base_config("fate_flow", {}).get("http_port")
GRPC_PORT = get_base_config("fate_flow", {}).get("grpc_port")

# standalone job will be send to the standalone job server when FATE-Flow work on cluster deploy mode,
# but not the port for FATE-Flow on standalone deploy mode.
CLUSTER_STANDALONE_JOB_SERVER_PORT = 9381


# services ip and port
SERVER_CONF_PATH = 'arch/conf/server_conf.json'
SERVING_PATH = '/servers/servings'
server_conf = file_utils.load_json_conf(SERVER_CONF_PATH)
PROXY_HOST = server_conf.get(SERVERS).get('proxy').get('host')
PROXY_PORT = server_conf.get(SERVERS).get('proxy').get('port')
BOARD_HOST = server_conf.get(SERVERS).get('fateboard').get('host')
if BOARD_HOST == 'localhost':
    BOARD_HOST = get_lan_ip()
BOARD_PORT = server_conf.get(SERVERS).get('fateboard').get('port')
MANAGER_HOST = server_conf.get(SERVERS).get('fatemanager', {}).get('host')
MANAGER_PORT = server_conf.get(SERVERS).get('fatemanager', {}).get('port')
SERVINGS = CenterConfig.get_settings(path=SERVING_PATH, servings_zk_path=SERVINGS_ZK_PATH,
                                     use_zk=USE_CONFIGURATION_CENTER, hosts=ZOOKEEPER_HOSTS,
                                     server_conf_path=SERVER_CONF_PATH)
BOARD_DASHBOARD_URL = 'http://%s:%d/index.html#/dashboard?job_id={}&role={}&party_id={}' % (BOARD_HOST, BOARD_PORT)

# switch
SAVE_AS_TASK_INPUT_DATA_SWITCH = True
SAVE_AS_TASK_INPUT_DATA_IN_MEMORY = True

# init
RuntimeConfig.init_config(WORK_MODE=WORK_MODE)
RuntimeConfig.init_config(HTTP_PORT=HTTP_PORT)
RuntimeConfig.init_config(BACKEND=BACKEND)
