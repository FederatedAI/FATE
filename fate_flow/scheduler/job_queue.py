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
import operator
import json
import threading
from time import monotonic as time

from fate_arch.common import base_utils
from fate_flow.db.db_models import DB, Job, NewQueue

from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.entity.constant import WorkMode, JobStatus
from fate_flow.settings import stat_logger, RE_ENTRY_QUEUE_MAX


class JobQueue(object):
    @classmethod
    def create_event(cls, job_id, initiator_role, initiator_party_id):
        with DB.connection_context():
            events = cls.query_event(job_id=job_id, initiator_role=initiator_role, initiator_party_id=initiator_party_id)
            if events:
                raise RuntimeError(f"job {job_id} {initiator_role} {initiator_party_id} already exists")
            event = NewQueue()
            event.f_job_id = job_id
            event.f_initiator_role = initiator_role
            event.f_initiator_party_id = initiator_party_id
            event.f_job_status = JobStatus
            event.f_create_time = base_utils.current_timestamp()
            event.save(force_insert=True)

    @classmethod
    def get_event(cls, status):
        events = cls.query_event(status=JobStatus.WAITING)
        return events

    @classmethod
    def update_event(cls, job_id, initiator_role, initiator_party_id, status):
        events = cls.query_event(job_id=job_id, initiator_role=initiator_role, initiator_party_id=initiator_party_id)
        if not events:
            raise RuntimeError(f"job {job_id} {initiator_role} {initiator_party_id} not in queue")
        event = events[0]
        if JobStatus.StateTransitionRule.if_pass(event.f_job_status, status):
            pass
        else:
            raise RuntimeError(f"can not update job status {event.f_job_status} to {status} on queue")

    @classmethod
    def query_event(cls, **kwargs):
        with DB.connection_context():
            query_filters = []
            for k, v in kwargs.items():
                attr_name = 'f_%s' % k
                if hasattr(NewQueue, attr_name):
                    query_filters.append(operator.attrgetter(attr_name)(NewQueue) == v)
            events = NewQueue.select().where(*query_filters)
            return [event for event in events]

    def qsize(self, status=None):
        if not status:
            status = 1
        with DB.connection_context():
            events = Queue.select().where(Queue.f_is_waiting == status)
            return len(events)


class ListQueue(BaseQueue):
    def __init__(self):
        super(ListQueue, self).__init__()
        self.queue = []
        self.ready = True
        self.mutex = threading.Lock()
        self.not_empty = threading.Condition(self.mutex)
        self.not_full = threading.Condition(self.mutex)
        self.maxsize = 0
        self.unfinished_tasks = 0
        stat_logger.info('init in-process queue')

    def put_event(self, event, status=None, job_id=None):
        try:
            self.put(event)
            stat_logger.info('put event into in-process queue successfully: {}'.format(event))
        except Exception as e:
            stat_logger.error('put event into in-process queue failed')
            stat_logger.exception(e)
            raise e

    def get_event(self, status=None, end_status=None):
        try:
            event = self.get(block=True)
            stat_logger.info('get event from in-process queue successfully: {}'.format(event))
            return event
        except Exception as e:
            stat_logger.error('get event from in-process queue failed')
            stat_logger.exception(e)
            return None

    def del_event(self, event):
        try:
            ret = self.dell(event)
            stat_logger.info('delete event from redis queue {}: {}'.format('successfully' if ret else 'failed', event))
        except Exception as e:
            stat_logger.info('delete event from  queue failed:{}'.format(str(e)))
            raise Exception('{} not in ListQueue'.format(event))

    def dell(self, event):
        with self.not_empty:
            if event in self.queue:
                self.queue.remove(event)
                self.not_full.notify()
            else:
                raise Exception('{} not in queue'.format(event))

    def put(self, item, block=True, timeout=None):
        with self.not_full:
            if self.maxsize > 0:
                if not block:
                    if self.qsize() >= self.maxsize:
                        raise Exception
                elif timeout is None:
                    while self.qsize() >= self.maxsize:
                        self.not_full.wait()
                elif timeout < 0:
                    raise ValueError("'timeout' must be a non-negative number")
                else:
                    endtime = time() + timeout
                    while self.qsize() >= self.maxsize:
                        remaining = endtime - time()
                        if remaining <= 0.0:
                            raise Exception
                        self.not_full.wait(remaining)
            self.queue.append(item)
            self.unfinished_tasks += 1
            self.not_empty.notify()

    def get(self, block=True, timeout=None):
        with self.not_empty:
            if not block:
                if not self.qsize():
                    raise Exception
            elif timeout is None:
                while not self.qsize():
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time() + timeout
                while not self.qsize():
                    remaining = endtime - time()
                    if remaining <= 0.0:
                        raise Exception
                    self.not_empty.wait(remaining)
            item = self.queue.pop(0)
            self.not_full.notify()
            return item

    def qsize(self, status=None):
        return len(self.queue)


def init_job_queue():
    if RuntimeConfig.WORK_MODE == WorkMode.STANDALONE:
        job_queue = ListQueue()
        RuntimeConfig.init_config(JOB_QUEUE=job_queue)
    elif RuntimeConfig.WORK_MODE == WorkMode.CLUSTER:
        job_queue = MysqlQueue()
        RuntimeConfig.init_config(JOB_QUEUE=job_queue)
    else:
        raise Exception('init queue failed.')

