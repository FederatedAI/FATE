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

from fate_flow.settings import stat_logger
from fate_flow.utils.api_utils import get_json_result, forward_api

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


@manager.route('/<role>', methods=['POST'])
def start_forward(role):
    request_config = request.json or request.form.to_dict()
    forward_api(job_id=request_config.get('body').get('job_id', None),
                method= request_config.get('header').get('method', 'POST'),
                endpoint=request_config.get('header').get('endpoint'),
                src_party_id=request_config.get('header').get('src_party_id'),
                dest_party_id=request_config.get('header').get('dest_party_id'),
                role=role,
                ip=request_config.get('header').get('id'),
                grpc_port=request_config.get('header').get('grpc_port'),
                json_body=request_config.get('body')
                )




