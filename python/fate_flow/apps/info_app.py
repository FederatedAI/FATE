#
#  Copyright 2021 The FATE Authors. All Rights Reserved.
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
from flask import Flask

from fate_arch.common.conf_utils import get_base_config
from fate_flow.settings import stat_logger
from fate_flow.utils.api_utils import error_response, get_json_result


manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return error_response(500, str(e))


@manager.route('/fateboard', methods=['POST'])
def get_fateboard_info():
    fateboard = get_base_config('fateboard', {})
    host = fateboard.get('host')
    port = fateboard.get('port')
    if not host or not port:
        return error_response(404, 'fateboard is not configured')
    return get_json_result(data={
        'host': host,
        'port': port,
    })
