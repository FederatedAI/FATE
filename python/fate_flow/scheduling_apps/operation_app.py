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

from flask import Flask, request

from fate_flow.entity.types import RetCode
from fate_flow.settings import stat_logger
from fate_flow.utils import job_utils
from fate_flow.utils.api_utils import get_json_result
from fate_arch.common import log, file_utils


manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=RetCode.EXCEPTION_ERROR, retmsg=log.exception_to_trace_string(e))


@manager.route('/job_config/get', methods=['POST'])
def get_config():
    request_data = request.json
    job_conf = job_utils.get_job_conf(request_data.get("job_id"), request_data.get("role"))
    return get_json_result(retcode=0, retmsg='success', data=job_conf)

@manager.route('/json_conf/load', methods=['POST'])
def load_json_conf():
    job_conf = file_utils.load_json_conf(request.json.get("config_path"))
    return get_json_result(retcode=0, retmsg='success', data=job_conf)
