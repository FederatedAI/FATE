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
from copy import deepcopy

import requests

from fate_arch.common.log import schedule_logger
from fate_flow.controller.engine_operation.base import BaseEngine
from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.entity.types import LinkisJobStatus, KillProcessStatusCode
from fate_flow.settings import LINKIS_SPARK_CONFIG, LINKIS_EXECUTE_ENTRANCE, LINKIS_SUBMIT_PARAMS, LINKIS_RUNTYPE, \
    LINKIS_LABELS, LINKIS_QUERT_STATUS, LINKIS_KILL_ENTRANCE, detect_logger


class LinkisSparkEngine(BaseEngine):
    @staticmethod
    def run(job_id, component_name, task_id, task_version, role, party_id, task_parameters_path, run_parameters,
            task_info, user_name, **kwargs):
        linkis_execute_url = "http://{}:{}{}".format(LINKIS_SPARK_CONFIG.get("host"),
                                                     LINKIS_SPARK_CONFIG.get("port"),
                                                     LINKIS_EXECUTE_ENTRANCE)
        headers = {"Token-Code": LINKIS_SPARK_CONFIG.get("token_code"),
                   "Token-User": user_name,
                   "Content-Type": "application/json"}
        schedule_logger(job_id).info(f"headers:{headers}")
        python_path = LINKIS_SPARK_CONFIG.get("python_path")
        execution_code = 'import sys\nsys.path.append("{}")\n' \
                         'from fate_flow.operation.task_executor import TaskExecutor\n' \
                         'task_info = TaskExecutor.run_task(job_id="{}",component_name="{}",' \
                         'task_id="{}",task_version={},role="{}",party_id={},' \
                         'run_ip="{}",config="{}",job_server="{}")\n' \
                         'TaskExecutor.report_task_update_to_driver(task_info=task_info)'. \
            format(python_path, job_id, component_name, task_id, task_version, role, party_id, RuntimeConfig.JOB_SERVER_HOST,
                   task_parameters_path, '{}:{}'.format(RuntimeConfig.JOB_SERVER_HOST, RuntimeConfig.HTTP_PORT))
        schedule_logger(job_id).info(f"execution code:{execution_code}")
        params = deepcopy(LINKIS_SUBMIT_PARAMS)
        schedule_logger(job_id).info(f"spark run parameters:{run_parameters.spark_run}")
        for spark_key, v in run_parameters.spark_run.items():
            if spark_key in ["spark.executor.memory", "spark.driver.memory", "spark.executor.instances",
                             "wds.linkis.rm.yarnqueue"]:
                params["configuration"]["startup"][spark_key] = v
        data = {
            "method": LINKIS_EXECUTE_ENTRANCE,
            "params": params,
            "executeApplicationName": "spark",
            "executionCode": execution_code,
            "runType": LINKIS_RUNTYPE,
            "source": {},
            "labels": LINKIS_LABELS
        }
        schedule_logger(job_id).info(f'submit linkis spark, data:{data}')
        task_info["engine_conf"]["data"] = data
        task_info["engine_conf"]["headers"] = headers
        res = requests.post(url=linkis_execute_url, headers=headers, json=data)
        schedule_logger(job_id).info(f"start linkis spark task: {res.text}")
        if res.status_code == 200:
            if res.json().get("status"):
                raise Exception(f"submit linkis spark failed: {res.json()}")
            task_info["engine_conf"]["execID"] = res.json().get("data").get("execID")
            task_info["engine_conf"]["taskID"] = res.json().get("data").get("taskID")
            schedule_logger(job_id).info('submit linkis spark success')
        else:
            raise Exception(f"submit linkis spark failed: {res.text}")
        return True

    @staticmethod
    def kill(task):
        linkis_query_url = "http://{}:{}{}".format(LINKIS_SPARK_CONFIG.get("host"),
                                                   LINKIS_SPARK_CONFIG.get("port"),
                                                   LINKIS_QUERT_STATUS.replace("execID",
                                                                               task.f_engine_conf.get("execID")))
        headers = task.f_engine_conf.get("headers")
        response = requests.get(linkis_query_url, headers=headers).json()
        schedule_logger(task.f_job_id).info(f"querty task response:{response}")
        if response.get("data").get("status") != LinkisJobStatus.SUCCESS:
            linkis_execute_url = "http://{}:{}{}".format(LINKIS_SPARK_CONFIG.get("host"),
                                                         LINKIS_SPARK_CONFIG.get("port"),
                                                         LINKIS_KILL_ENTRANCE.replace("execID",
                                                                                      task.f_engine_conf.get("execID")))
            schedule_logger(task.f_job_id).info(f"start stop task:{linkis_execute_url}")
            schedule_logger(task.f_job_id).info(f"headers: {headers}")
            kill_result = requests.get(linkis_execute_url, headers=headers)
            schedule_logger(task.f_job_id).info(f"kill result:{kill_result}")
            if kill_result.status_code == 200:
                pass
        return KillProcessStatusCode.KILLED

    @staticmethod
    def is_alive(task):
        process_exist = True
        try:
            linkis_query_url = "http://{}:{}{}".format(LINKIS_SPARK_CONFIG.get("host"),
                                                       LINKIS_SPARK_CONFIG.get("port"),
                                                       LINKIS_QUERT_STATUS.replace("execID", task.f_engine_conf.get("execID")))
            headers = task.f_engine_conf["headers"]
            response = requests.get(linkis_query_url, headers=headers).json()
            detect_logger.info(response)
            if response.get("data").get("status") == LinkisJobStatus.FAILED:
                process_exist = False
        except Exception as e:
            detect_logger.exception(e)
            process_exist = False
        return process_exist