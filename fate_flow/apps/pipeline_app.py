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
from fate_flow.utils.api_utils import get_json_result
from fate_flow.settings import stat_logger
from flask import Flask, request
from fate_flow.manager import pipeline_manager

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


@manager.route('/dag/dependency', methods=['post'])
def pipeline_dag_dependency():
    dependency = pipeline_manager.pipeline_dag_dependency(request.json)
    if dependency:
        return get_json_result(retcode=0, retmsg='success', data=dependency)
    else:
        return get_json_result(retcode=101, retmsg='')
