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
import json
from arch.api import eggroll
from arch.api.utils import file_utils, dtable_utils
from arch.task_manager.adapter.offline_feature.get_feature import GetFeature
from arch.task_manager.job_manager import save_job_info, query_job_by_id, generate_job_id, get_job_directory, new_runtime_conf, run_subprocess
from arch.task_manager.utils.api_utils import get_json_result, local_api, new_federated_job
from arch.task_manager.settings import logger, PARTY_ID
from flask import Flask, request
import datetime
import os
from arch.task_manager.settings import WORK_MODE, JOB_MODULE_CONF

manager = Flask(__name__)


@manager.errorhandler(500)
def internal_server_error(e):
    logger.exception(e)
    return get_json_result(status=100, msg=str(e))


@manager.route('/<data_func>', methods=['post'])
def download_upload(data_func):
    return new_federated_job(request)


@manager.route('/<data_func>/do', methods=['POST'])
def do_download_upload(data_func):
    request_config = request.json
    _job_id = generate_job_id()
    logger.info('generated job_id {}, body {}'.format(_job_id, request_config))
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
            return get_json_result(status=102, msg='no table name and namespace')
        request_config['table_name'] = table_name
        request_config['namespace'] = namespace
        conf_file_path = new_runtime_conf(job_dir=_job_dir, method=data_func, module=module,
                                          role=request_config.get('local', {}).get("role"),
                                          party_id=request_config.get('local', {}).get("party_id", PARTY_ID))
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
        p = run_subprocess(job_dir=_job_dir, job_role=data_func, progs=progs)
        return get_json_result(job_id=_job_id, data={'pid': p.pid, 'table_name': request_config['table_name'], 'namespace': request_config['namespace']})
    except Exception as e:
        logger.exception(e)
        return get_json_result(status=-104, msg="failed", job_id=_job_id)


@manager.route('/importIdFromLocal', methods=['POST'])
def import_id_from_local():
    return new_federated_job(request)


@manager.route('/importIdFromLocal/do', methods=['POST'])
def do_import_id_from_local():
    request_config = request.json
    if not os.path.isabs(request_config.get("file", "")):
        input_file_path = os.path.join(file_utils.get_project_base_directory(), request_config["file"])
    else:
        input_file_path = request_config["file"]
    batch_size = request_config.get("batch_size", 10)
    post_data = {}
    is_success = True
    response_msg = ""
    with open(input_file_path) as fr:
        id_tmp = []
        range_start = 0
        range_end = -1
        total = 0
        file_end = False
        while True:
            for i in range(batch_size):
                line = fr.readline()
                if not line:
                    file_end = True
                    break
                id_tmp.append(line.split(",")[0])
                range_end += 1
                total += 1
            post_data["rangeStart"] = range_start
            post_data["rangeEnd"] = range_end
            post_data["ids"] = id_tmp
            if file_end:
                # file end
                post_data["total"] = total
                response = local_api(method='POST',
                                     suffix='/data/importId',
                                     json_body=post_data)
                response_msg = response.get("msg")
                break
            else:
                post_data["total"] = 0
                response = local_api(method='POST',
                                     suffix='/data/importId',
                                     json_body=post_data)
                if response.get("status") != 0:
                    is_success = False
                    break
            range_start = range_end + 1
            del id_tmp[:]
    if is_success:
        return get_json_result(msg=response_msg)
    else:
        return get_json_result(status=101, msg="import id error")


@manager.route('/importId', methods=['POST'])
def import_id():
    eggroll.init(job_id=generate_job_id(), mode=WORK_MODE)
    request_data = request.json
    namespace = "id_library"
    id_library_info = eggroll.table("info", namespace, partition=10, create_if_missing=True, error_if_exist=False)
    if request_data.get("rangeStart") == 0:
        data_id = generate_job_id()
        id_library_info.put("tmp_data_id", data_id)
    else:
        data_id = id_library_info.get("tmp_data_id")
    data_table = eggroll.table(data_id, namespace, partition=50, create_if_missing=True, error_if_exist=False)
    for i in request_data.get("ids", []):
        data_table.put(i, "")
    if request_data.get("rangeEnd") and request_data.get("total") and (request_data.get("total") - request_data.get("rangeEnd") == 1):
        # end
        new_id_count = data_table.count()
        if new_id_count == request_data["total"]:
            id_library_info.put(data_id, json.dumps({"salt": request_data.get("salt"), "saltMethod": request_data.get("saltMethod")}))
            old_data_id = id_library_info.get("use_data_id")
            id_library_info.put("use_data_id", data_id)
            logger.info("import id success, dtable name is {}, namespace is {}".format(data_id, namespace))

            # TODO: destroy DTable, should be use a lock
            try:
                old_data_table = eggroll.table(old_data_id, namespace, create_if_missing=True, error_if_exist=False)
                old_data_table.destroy()
            except Exception as e:
                logger.warning(e)
            id_library_info.delete(old_data_id)
        else:
            data_table.destroy()
            return get_json_result(102, "the actual amount of data is not equal to total.")
    return get_json_result(data={'table_name': data_id, 'namespace': namespace})


@manager.route('/requestOfflineFeature', methods=['POST'])
def request_offline_feature():
    return new_federated_job(request)


@manager.route('/requestOfflineFeature/do', methods=['POST'])
def do_request_offline_feature():
    request_config = request.json
    job_id = generate_job_id()
    response = GetFeature.request(job_id, request_config)
    if response.get("status", 1) == 0:
        job_data = dict()
        job_data.update(request_config)
        job_data["begin_date"] = datetime.datetime.now()
        job_data["status"] = "running"
        job_data["config"] = json.dumps(request_config)
        save_job_info(job_id=job_id,
                      role=request_config.get('local', {}).get("role"),
                      party_id=request_config.get('local', {}).get("party_id"),
                      save_info=job_data)
        return get_json_result(job_id=job_id, msg="request offline feature successfully")
    else:
        return get_json_result(status=101, msg="request offline feature error: %s" % response.get("msg", ""))


@manager.route('/importOfflineFeature', methods=['POST'])
def import_offline_feature():
    eggroll.init(job_id=generate_job_id(), mode=WORK_MODE)
    request_data = request.json
    if not request_data.get("jobId"):
        return get_json_result(status=2, msg="no job id")
    _job_id = request_data.get("jobId")
    _job_dir = get_job_directory(job_id=_job_id)
    jobs = query_job_by_id(job_id=_job_id)
    if not jobs:
        return get_json_result(status=3, msg="can not found this job id: %s" % request_data.get("jobId", ""))
    job_config = json.loads(jobs[0].config)
    job_config.update(request_data)
    job_config["work_mode"] = WORK_MODE
    module = "importFeature"
    conf_file_path = new_runtime_conf(job_dir=_job_dir, method="import", module=module,
                                      role=job_config.get('local', {}).get("role"),
                                      party_id=job_config.get('local', {}).get("party_id", PARTY_ID))
    file_utils.dump_json_conf(job_config, conf_file_path)
    progs = ["python3",
             os.path.join(file_utils.get_project_base_directory(), JOB_MODULE_CONF[module]["module_path"]),
             "-j", _job_id,
             "-c", conf_file_path
             ]
    p = run_subprocess(job_dir=_job_dir, job_role="import", progs=progs)
    return get_json_result(job_id=_job_id)
