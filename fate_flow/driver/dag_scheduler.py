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
import threading
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from arch.api.utils.log_utils import schedule_logger
from fate_flow.driver.job_controller import TaskScheduler
from fate_flow.manager.queue_manager import BaseQueue




class DAGScheduler(threading.Thread):
    def __init__(self, queue: BaseQueue, concurrent_num: int = 1):
        super(DAGScheduler, self).__init__()
        self.concurrent_num = concurrent_num
        self.queue = queue
        self.job_executor_pool = ThreadPoolExecutor(max_workers=concurrent_num)

    def run(self):
        if not self.queue.is_ready():
            schedule_logger().error('queue is not ready')
            return False
        all_jobs = []
        while True:
            try:
                if len(all_jobs) == self.concurrent_num:
                    for future in as_completed(all_jobs):
                        all_jobs.remove(future)
                        break
                job_event = self.queue.get_event()
                schedule_logger(job_event['job_id']).info('schedule job {}'.format(job_event))
                future = self.job_executor_pool.submit(DAGScheduler.handle_event, job_event)
                future.add_done_callback(DAGScheduler.get_result)
                all_jobs.append(future)
            except Exception as e:
                schedule_logger().exception(e)

    def stop(self):
        self.job_executor_pool.shutdown(True)

    @staticmethod
    def handle_event(job_event):
        try:
            return TaskScheduler.run_job(**job_event)
        except Exception as e:
            schedule_logger(job_event.get('job_id')).exception(e)
            return False

    @staticmethod
    def get_result(future):
        future.result()
