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
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

from fate_flow.entity.constant_config import JobStatus, WorkMode

from arch.api.utils.log_utils import schedule_logger
from fate_flow.driver.job_controller import TaskScheduler
from fate_flow.manager.queue_manager import BaseQueue
from fate_flow.settings import RE_ENTRY_QUEUE_TIME, stat_logger, WORK_MODE


class DAGScheduler(threading.Thread):
    def __init__(self, queue: BaseQueue, concurrent_num: int = 1):
        super(DAGScheduler, self).__init__()
        self.concurrent_num = concurrent_num
        self.queue = queue
        self.job_executor_pool = ThreadPoolExecutor(max_workers=concurrent_num)

    def run(self):
        time.sleep(5)
        if not self.queue.is_ready():
            schedule_logger().error('queue is not ready')
            return False
        all_jobs = []

        if WORK_MODE == WorkMode.CLUSTER:
            # get Mediation queue job
            t = threading.Thread(target=mediation_queue_put_events, args=[self.queue])
            t.start()

            t = threading.Thread(target=queue_put_events, args=[self.queue])
            t.start()

        schedule_logger().info('start get event from queue')
        while True:
            try:
                if len(all_jobs) == self.concurrent_num:
                    for future in as_completed(all_jobs):
                        all_jobs.remove(future)
                        break
                job_event = self.queue.get_event(end_status=5)
                schedule_logger(job_event['job_id']).info('start check all role status')
                status = TaskScheduler.check(job_event['job_id'], job_event['initiator_role'], job_event['initiator_party_id'])
                schedule_logger(job_event['job_id']).info('check all role status success, status is {}'.format(status))
                if not status:
                    TaskScheduler.cancel_ready(job_event['job_id'], job_event['initiator_role'], job_event['initiator_party_id'])
                    schedule_logger(job_event['job_id']).info('host is busy, job {} into waiting......'.format(job_event['job_id']))
                    is_failed = self.queue.put_event(job_event, status=3, job_id=job_event['job_id'])
                    schedule_logger(job_event['job_id']).info('job into queue_2 status is {}'.format('success' if not is_failed else 'failed'))
                    if is_failed:
                        schedule_logger(job_event['job_id']).info('start to cancel job')
                        TaskScheduler.stop(job_id=job_event['job_id'], end_status=JobStatus.CANCELED)
                else:
                    self.queue.set_status(job_id=job_event['job_id'], status=0)
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


def queue_put_events(queue):
    while True:
        n = queue.qsize(status=3)
        stat_logger.info('start run queue_2, total num {}'.format(n))
        for i in range(n):
            event = queue.get_event(status=3)
            is_failed = queue.put_event(event, job_id=event['job_id'], status=1)
            schedule_logger(event['job_id']).info('job into queue_1 status is {}'.format('success' if not is_failed else 'failed'))
            if is_failed:
                schedule_logger(event['job_id']).info('start to cancel job')
                try:
                    TaskScheduler.stop(job_id=event['job_id'], end_status=JobStatus.CANCELED)
                except Exception as e:
                    schedule_logger(event['job_id']).info('cancel failed:{}'.format(e))
        time.sleep(RE_ENTRY_QUEUE_TIME)


def mediation_queue_put_events(queue):
    n = queue.qsize(status=5)
    stat_logger.info('start check mediation queue, total num {}'.format(n))
    for i in range(n):
        event = queue.get_event(status=5)
        try:
            TaskScheduler.cancel_ready(event['job_id'], event['initiator_role'], event['initiator_party_id'])
            is_failed = queue.put_event(event, job_id=event['job_id'], status=1)
            schedule_logger(event['job_id']).info('job into queue_1 status is {}'.format('success' if not is_failed else 'failed'))
            if is_failed:
                schedule_logger(event['job_id']).info('start to cancel job')
                TaskScheduler.stop(job_id=event['job_id'], end_status=JobStatus.CANCELED)
        except Exception as e:
            schedule_logger(event['job_id']).error(e)
            try:
                schedule_logger(event['job_id']).info('start cancel job')
                TaskScheduler.stop(job_id=event['job_id'], end_status=JobStatus.CANCELED)
            except:
                schedule_logger(event['job_id']).info('cancel job failed')





