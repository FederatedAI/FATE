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
from arch.api.utils import file_utils
from arch.task_manager.adapter.offline_feature.get_feature import GetFeature
from arch.task_manager.job_manager import save_job_info, query_job_by_id, update_job_by_id, generate_job_id, get_job_directory, get_json_result
from arch.task_manager.settings import logger
from flask import Flask, request
import datetime
import os
import subprocess
import uuid
from arch.task_manager.settings import WORK_MODE

manager = Flask(__name__)


@manager.route('/<data_func>', methods=['post'])
def download_data(data_func):
    _data = request.json
    _job_id = generate_job_id()
    logger.info('generated job_id {}, body {}'.format(_job_id, _data))
    _job_dir = get_job_directory(_job_id)
    os.makedirs(_job_dir, exist_ok=True)
    _download_module = os.path.join(file_utils.get_project_base_directory(), "arch/api/utils/download.py")
    _upload_module = os.path.join(file_utils.get_project_base_directory(), "arch/api/utils/upload.py")

    if data_func == "download":
        _module = _download_module
    else:
        _module = _upload_module

    try:
        if data_func == "download":
            progs = ["python3",
                     _module,
                     "-j", _job_id,
                     "-c", os.path.abspath(_data.get("config_path"))
                     ]
        else:
            progs = ["python3",
                     _module,
                     "-c", os.path.abspath(_data.get("config_path"))
                     ]

        logger.info('Starting progs: {}'.format(progs))

        std_log = open(os.path.join(_job_dir, 'std.log'), 'w')
        task_pid_path = os.path.join(_job_dir, 'pids')

        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
        else:
            startupinfo = None
        p = subprocess.Popen(progs,
                             stdout=std_log,
                             stderr=std_log,
                             startupinfo=startupinfo
                             )

        os.makedirs(task_pid_path, exist_ok=True)
        with open(os.path.join(task_pid_path, data_func + ".pid"), 'w') as f:
            f.truncate()
            f.write(str(p.pid) + "\n")
            f.flush()

        return get_json_result(0, "success, job_id {}".format(_job_id))
    except Exception as e:
        print(e)
        return get_json_result(-104, "failed, job_id {}".format(_job_id))


@manager.route('/importId', methods=['POST'])
def import_id():
    eggroll.init(job_id=generate_job_id(), mode=WORK_MODE)
    request_data = request.json
    table_name_space = "id_library"
    try:
        id_library_info = eggroll.table("info", table_name_space, partition=10, create_if_missing=True, error_if_exist=False)
        if request_data.request("rangeStart") == 0:
            data_id = generate_job_id()
            id_library_info.put("tmp_data_id", data_id)
        else:
            data_id = id_library_info.request("tmp_data_id")
        data_table = eggroll.table(data_id, table_name_space, partition=50, create_if_missing=True, error_if_exist=False)
        for i in request_data.request("ids", []):
            data_table.put(i, "")
        if request_data.request("rangeEnd") and request_data.request("total") and (request_data.request("total") - request_data.request("rangeEnd") == 1):
            # end
            new_id_count = data_table.count()
            if new_id_count == request_data["total"]:
                id_library_info.put(data_id, json.dumps({"salt": request_data.request("salt"), "saltMethod": request_data.request("saltMethod")}))
                old_data_id = id_library_info.request("use_data_id")
                id_library_info.put("use_data_id", data_id)
                logger.info("import id success, dtable name is {}, namespace is {}", data_id, table_name_space)

                # TODO: destroy DTable, should be use a lock
                old_data_table = eggroll.table(old_data_id, table_name_space, partition=50, create_if_missing=True, error_if_exist=False)
                old_data_table.destroy()
                id_library_info.delete(old_data_id)
            else:
                data_table.destroy()
                return get_json_result(2, "the actual amount of data is not equal to total.")
        return get_json_result()
    except Exception as e:
        logger.exception(e)
        return get_json_result(1, "import error.")


@manager.route('/requestOfflineFeature', methods=['POST'])
def request_offline_feature():
    request_data = request.json
    try:
        job_id = uuid.uuid1().hex
        response = GetFeature.request(job_id, request_data)
        if response.get("status", 1) == 0:
            job_data = dict()
            job_data.update(request_data)
            job_data["begin_date"] = datetime.datetime.now()
            job_data["status"] = "running"
            job_data["config"] = json.dumps(request_data)
            save_job_info(job_id=job_id, **job_data)
            return get_json_result()
        else:
            return get_json_result(status=1, msg="request offline feature error: %s" % response.get("msg", ""))
    except Exception as e:
        logger.exception(e)
        return get_json_result(status=1, msg="request offline feature error: %s" % e)


@manager.route('/importOfflineFeature', methods=['POST'])
def import_offline_feature():
    eggroll.init(job_id=generate_job_id(), mode=WORK_MODE)
    request_data = request.json
    try:
        if not request_data.get("jobId"):
            return get_json_result(status=2, msg="no job id")
        job_id = request_data.get("jobId")
        job_data = query_job_by_id(job_id=job_id)
        if not job_data:
            return get_json_result(status=3, msg="can not found this job id: %s" % request_data.get("jobId", ""))
        response = GetFeature.import_data(request_data, json.loads(job_data[0]["config"]))
        if response.get("status", 1) == 0:
            update_job_by_id(job_id=job_id, update_data={"status": "success", "end_date": datetime.datetime.now()})
            return get_json_result()
        else:
            return get_json_result(status=1, msg="request offline feature error: %s" % response.get("msg", ""))
    except Exception as e:
        logger.exception(e)
        return get_json_result(status=1, msg="request offline feature error: %s" % e)
