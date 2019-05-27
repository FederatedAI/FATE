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
from arch.api.utils import file_utils
from arch.api.utils.log_utils import LoggerFactory
LoggerFactory.setDirectory()
logger = LoggerFactory.getLogger("task_manager")

'''
Constants
'''

API_VERSION = "v1"
ROLE = 'manager'
SERVERS = 'servers'
MAX_CONCURRENT_JOB_RUN = 5
DEFAULT_WORKFLOW_DATA_TYPE = ['train_input', 'data_input', 'id_library_input', 'model', 'predict_input', 'predict_output', 'evaluation_output', 'intersect_data_output']
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
DEFAULT_GRPC_OVERALL_TIMEOUT = 60 * 1000  # ms
HEADERS = {
    'Content-Type': 'application/json',
}
JOB_SCHEDULER = True


IP = '0.0.0.0'
GRPC_PORT = 9360
HTTP_PORT = 9380
PARTY_ID = 9999
WORK_MODE = 0
LOCAL_URL = "http://localhost:{}".format(HTTP_PORT)

DATABASE = {
    'engine': 'mysql',
    'name': 'task_manager',
    'user': 'root',
    'passwd': 'root1234',
    'host': '127.0.0.1',
    'port': 3306,
    'max_connections': 100,
    'stale_timeout': 30,
}

server_conf = file_utils.load_json_conf("arch/conf/server_conf.json")
PROXY_HOST = server_conf.get(SERVERS).get('proxy').get('host')
PROXY_PORT = server_conf.get(SERVERS).get('proxy').get('port')
SERVINGS = server_conf.get(SERVERS).get('servings')
JOB_MODULE_CONF = file_utils.load_json_conf("arch/task_manager/job_module_conf.json")
