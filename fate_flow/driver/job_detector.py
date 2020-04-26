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
from arch.api.utils.core_utils import get_lan_ip, json_loads
from arch.api.utils.log_utils import schedule_logger
from fate_flow.driver.task_scheduler import TaskScheduler
from fate_flow.settings import detect_logger, API_VERSION
from fate_flow.utils import cron, job_utils, api_utils


class JobDetector(cron.Cron):
    def run_do(self):
        try:
            running_tasks = job_utils.query_task(status='running', run_ip=get_lan_ip())
            stop_job_ids = set()
            # detect_logger.info('start to detect running job..')
            for task in running_tasks:
                try:
                    process_exist = job_utils.check_job_process(int(task.f_run_pid))
                    if not process_exist:
                        detect_logger.info(
                            'job {} component {} on {} {} task {} {} process does not exist'.format(task.f_job_id,
                                                                                                    task.f_component_name,
                                                                                                    task.f_role,
                                                                                                    task.f_party_id,
                                                                                                    task.f_task_id,
                                                                                                    task.f_run_pid))
                        stop_job_ids.add(task.f_job_id)
                except Exception as e:
                    detect_logger.exception(e)
            if stop_job_ids:
                schedule_logger().info('start to stop jobs: {}'.format(stop_job_ids))
            for job_id in stop_job_ids:
                jobs = job_utils.query_job(job_id=job_id)
                if jobs:
                    initiator_party_id = jobs[0].f_initiator_party_id
                    job_work_mode = jobs[0].f_work_mode
                    if len(jobs) > 1:
                        # i am initiator
                        my_party_id = initiator_party_id
                    else:
                        my_party_id = jobs[0].f_party_id
                        initiator_party_id = jobs[0].f_initiator_party_id
                    api_utils.federated_api(job_id=job_id,
                                            method='POST',
                                            endpoint='/{}/job/stop'.format(
                                                API_VERSION),
                                            src_party_id=my_party_id,
                                            dest_party_id=initiator_party_id,
                                            src_role=None,
                                            json_body={'job_id': job_id},
                                            work_mode=job_work_mode)
                    TaskScheduler.finish_job(job_id=job_id, job_runtime_conf=json_loads(jobs[0].f_runtime_conf), stop=True)
        except Exception as e:
            detect_logger.exception(e)
        finally:
            detect_logger.info('finish detect running job')
