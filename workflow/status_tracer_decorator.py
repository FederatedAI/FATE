#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

from arch.api.utils import file_utils
from arch.api.utils import log_utils
from arch.api import session
import argparse
import requests
import traceback

LOGGER = log_utils.getLogger()


server_conf = file_utils.load_json_conf("arch/conf/server_conf.json")
SERVERS = "servers"
ROLE = "manager"
IP = server_conf.get(SERVERS).get(ROLE).get("host")
HTTP_PORT = server_conf.get(SERVERS).get(ROLE).get("http.port")
LOCAL_URL = "http://{}:{}".format(IP, HTTP_PORT) + "/job/jobStatus"
job_id = None


def call_back(status):
    global job_id
    global role
    global party_id
    global LOCAL_URL

    if job_id is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('-j', '--job_id', type=str, required=True, help="Specify the jobid")
        parser.add_argument('-c', '--config', required=True, type=str, help="Specify a config json file path")

        args = parser.parse_args()
        job_id = args.job_id
        config = file_utils.load_json_conf(args.config)
        role = config.get('local', {}).get('role')
        party_id = config.get('local', {}).get('party_id')

    try:
        requests.post("/".join([LOCAL_URL, str(job_id), str(role), str(party_id)]), json={"status": status})
    except:
        LOGGER.info("fail to post status {}".format(status))


def status_trace(func):

    def wrapper(self, *args, **kwargs):
        call_back("running")
        
        res = None
        try:
            res = func(self, *args, **kwargs)
            call_back("success")
            LOGGER.info("job status is success")
        except Exception as e:
            LOGGER.info("job status is failed")
            LOGGER.info("{}".format(traceback.format_exc()))
            call_back("failed")

        return res

    return wrapper
