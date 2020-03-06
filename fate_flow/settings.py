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

from arch.api.utils import file_utils
from arch.api.utils import log_utils
from fate_flow.entity.runtime_config import RuntimeConfig
from arch.api.utils.core import get_lan_ip
import __main__

from fate_flow.utils.setting_utils import CenterConfig


WORK_MODE = 0
USE_LOCAL_DATABASE = True

# Local authentication switch
USE_AUTHENTICATION = False
PRIVILEGE_COMMAND_WHITELIST = []

# Node check switch
CHECK_NODES_IDENTITY = False

# zookeeper
USE_CONFIGURATION_CENTER = False
ZOOKEEPER_HOSTS = ['127.0.0.1:2181']

MAX_CONCURRENT_JOB_RUN = 5
MAX_CONCURRENT_JOB_RUN_HOST = 10
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
DEFAULT_GRPC_OVERALL_TIMEOUT = 60 * 1000 * 20  # ms
JOB_DEFAULT_TIMEOUT = 7 * 24 * 60 * 60
REDIS_QUEUE_DB_INDEX = 0

DATABASE = {
    'name': 'fate_flow',
    'user': 'root',
    'passwd': 'fate_dev',
    'host': '127.0.0.1',
    'port': 3306,
    'max_connections': 100,
    'stale_timeout': 30,
}

REDIS = {
    'host': '127.0.0.1',
    'port': 6379,
    'password': 'fate_dev',
    'max_connections': 500
}

'''
Constants
'''
API_VERSION = "v1"
ROLE = 'fateflow'
SERVERS = 'servers'
MAIN_MODULE = os.path.relpath(__main__.__file__)
SERVER_MODULE = 'fate_flow_server.py'
TASK_EXECUTOR_MODULE = 'driver/task_executor.py'
DEFAULT_WORKFLOW_DATA_TYPE = ['train_input', 'data_input', 'id_library_input', 'model', 'predict_input',
                              'predict_output', 'evaluation_output', 'intersect_data_output']
HEADERS = {
    'Content-Type': 'application/json',
    'Connection': 'close'
}
# fate-serving
SERVINGS_ZK_PATH = '/FATE-SERVICES/serving/online/publishLoad/providers'
FATE_FLOW_ZK_PATH = '/FATE-SERVICES/flow/online/transfer/providers'
FATE_FLOW_MODEL_TRANSFER_PATH = '/v1/model/transfer'
# fate-manager
FATE_MANAGER_GET_NODE_INFO = '/node/info'
FATE_MANAGER_NODE_CHECK = '/node/management/check'

# logger
log_utils.LoggerFactory.set_directory(os.path.join(file_utils.get_project_base_directory(), 'logs', 'fate_flow'))
stat_logger = log_utils.getLogger("fate_flow_stat")
detect_logger = log_utils.getLogger("fate_flow_detect")
access_logger = log_utils.getLogger("fate_flow_access")


"""
Services 
"""
IP = '0.0.0.0'
GRPC_PORT = 9360
HTTP_PORT = 9380

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
RuntimeConfig.init_config(WORK_MODE=WORK_MODE)
RuntimeConfig.init_config(HTTP_PORT=HTTP_PORT)
