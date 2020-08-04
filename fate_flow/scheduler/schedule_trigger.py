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
from arch.api.utils.log_utils import schedule_logger
from fate_flow.scheduler.dag_scheduler import DAGScheduler
from fate_flow.utils import cron, job_utils
from fate_flow.entity.constant import JobStatus


class ScheduleTrigger(cron.Cron):
    def run_do(self):
        try:
            schedule_logger().info("Start checking for scheduling events")
            running_jobs = job_utils.query_job(status=JobStatus.RUNNING, is_initiator=1)
            for job in running_jobs:
                try:
                    DAGScheduler.schedule(job)
                except Exception as e:
                    schedule_logger(job_id=job.f_job_id).exception(e)
        except Exception as e:
            schedule_logger().exception(e)
        finally:
            schedule_logger().info("Finish scheduled event check")
