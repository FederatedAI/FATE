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
from flask import Flask, request

from fate_arch.common.file_utils import get_federatedml_setting_conf_directory
from fate_flow.settings import stat_logger
from fate_flow.utils.api_utils import error_response, get_json_result
from fate_flow.utils.detect_utils import check_config
from fate_flow.scheduler.dsl_parser import DSLParser, DSLParserV2


manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return error_response(500, str(e))


@manager.route('/validate', methods=['POST'])
def validate_component_param():
    try:
        check_config(request.json, [
            'role',
            'component_parameters',
            'component_name',
            'component_module_name',
        ])
    except Exception as e:
        return error_response(400, str(e))

    args = [
        get_federatedml_setting_conf_directory(),
        {
            'role': request.json['role'],
            'component_parameters': request.json['component_parameters'],
        },
        request.json['component_name'],
        request.json['component_module_name'],
    ]
    try:
        if int(request.json.get('dsl_version', 1)) == 2:
            DSLParserV2.validate_component_param(*args)
        else:
            DSLParser.validate_component_param(*args)
    except Exception as e:
        return error_response(400, str(e))

    return get_json_result()
