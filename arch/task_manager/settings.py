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

'''
Constants
'''

API_VERSION = "v1"
ROLE = 'manager'
SERVERS = 'servers'
MAX_CONCURRENT_JOB_RUN = 5

_ONE_DAY_IN_SECONDS = 60 * 60 * 24
HEADERS = {
    'Content-Type': 'application/json',
}


server_conf = file_utils.load_json_conf("arch/conf/server_conf.json")
IP = server_conf.get(SERVERS).get(ROLE).get('host')
GRPC_PORT = server_conf.get(SERVERS).get(ROLE).get('grpc.port')
HTTP_PORT = server_conf.get(SERVERS).get(ROLE).get('http.port')
LOCAL_URL = "http://{}:{}".format(IP, HTTP_PORT)
PROXY_HOST = server_conf.get(SERVERS).get('proxy').get('host')
PROXY_PORT = server_conf.get(SERVERS).get('proxy').get('port')
WORK_MODE = server_conf.get(SERVERS).get(ROLE).get('work_mode')
PARTY_ID = server_conf.get('party_id')

DATABASE = {
    'engine': 'mysql',
    'name': 'task_manager_%d' % PARTY_ID,
    'user': 'root',
    'passwd': 'root1234',
    'host': '127.0.0.1',
    'port': 3306,
    'max_connections': 100,
    'stale_timeout': 30,
}
