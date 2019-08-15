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
import os

from flask import Flask, request

from arch.api.utils import file_utils
from fate_flow.settings import JOB_MODULE_CONF
from fate_flow.settings import stat_logger
from fate_flow.utils.api_utils import get_json_result
from fate_flow.utils.job_utils import generate_job_id, get_job_directory, new_runtime_conf, run_subprocess
from fate_flow.utils import detect_utils

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    stat_logger.exception(e)
    return get_json_result(retcode=100, retmsg=str(e))


@manager.route('/<data_func>', methods=['post'])
def download_upload(data_func):
    request_config = request.json
    _job_id = generate_job_id()
    stat_logger.info('generated job_id {}, body {}'.format(_job_id, request_config))
    _job_dir = get_job_directory(_job_id)
    os.makedirs(_job_dir, exist_ok=True)
    module = data_func
    required_arguments = ['work_mode', 'namespace', 'table_name']
    if module == 'upload':
        required_arguments.extend(['file', 'head', 'partition'])
    elif module == 'download':
        required_arguments.extend(['output_path'])
    else:
        raise Exception('can not support this operating: {}'.format(module))
    detect_utils.check_config(request_config, required_arguments=required_arguments)
    if module == "upload":
        if not os.path.isabs(request_config['file']):
            request_config["file"] = os.path.join(file_utils.get_project_base_directory(), request_config["file"])
    try:
        conf_file_path = new_runtime_conf(job_dir=_job_dir, method=data_func, module=module,
                                          role=request_config.get('local', {}).get("role"),
                                          party_id=request_config.get('local', {}).get("party_id", ''))
        file_utils.dump_json_conf(request_config, conf_file_path)
        progs = ["python3",
                 os.path.join(file_utils.get_project_base_directory(), JOB_MODULE_CONF[module]["module_path"]),
                 "-j", _job_id,
                 "-c", conf_file_path
                 ]
        try:
            p = run_subprocess(config_dir=_job_dir, process_cmd=progs)
        except Exception as e:
            stat_logger.exception(e)
            p = None
        return get_json_result(retcode=(0 if p else 101), job_id=_job_id,
                               data={'table_name': request_config['table_name'],
                                     'namespace': request_config['namespace'], 'pid': p.pid if p else ''})
    except Exception as e:
        stat_logger.exception(e)
        return get_json_result(retcode=-104, retmsg="failed", job_id=_job_id)
