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
from arch.api.utils import file_utils, dtable_utils
from fate_flow.utils.job_utils import generate_job_id, get_job_directory, new_runtime_conf, run_subprocess
from fate_flow.utils.api_utils import get_json_result
from fate_flow.settings import stat_logger
from flask import Flask, request
import os
from fate_flow.settings import WORK_MODE, JOB_MODULE_CONF

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
    if module == "upload":
        if not os.path.isabs(request_config.get("file", "")):
            request_config["file"] = os.path.join(file_utils.get_project_base_directory(), request_config["file"])
    try:
        request_config["work_mode"] = request_config.get('work_mode', WORK_MODE)
        table_name, namespace = dtable_utils.get_table_info(config=request_config, create=(True if module == 'upload' else False))
        if not table_name or not namespace:
            return get_json_result(retcode=102, retmsg='no table name and namespace')
        request_config['table_name'] = table_name
        request_config['namespace'] = namespace
        conf_file_path = new_runtime_conf(job_dir=_job_dir, method=data_func, module=module,
                                          role=request_config.get('local', {}).get("role"),
                                          party_id=request_config.get('local', {}).get("party_id", ''))
        file_utils.dump_json_conf(request_config, conf_file_path)
        if module == "download":
            progs = ["python3",
                     os.path.join(file_utils.get_project_base_directory(), JOB_MODULE_CONF[module]["module_path"]),
                     "-j", _job_id,
                     "-c", conf_file_path
                     ]
        else:
            progs = ["python3",
                     os.path.join(file_utils.get_project_base_directory(), JOB_MODULE_CONF[module]["module_path"]),
                     "-c", conf_file_path
                     ]
        p = run_subprocess(config_dir=_job_dir, process_cmd=progs)
        return get_json_result(job_id=_job_id, data={'pid': p.pid, 'table_name': request_config['table_name'], 'namespace': request_config['namespace']})
    except Exception as e:
        stat_logger.exception(e)
        return get_json_result(retcode=-104, retmsg="failed", job_id=_job_id)
