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
import operator
import queue
import threading
from time import monotonic as time

import redis
from fate_flow.db.db_models import DB, Job, Queue

from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.entity.constant_config import WorkMode
from fate_flow.settings import stat_logger


class BaseQueue:
    def __init__(self):
        self.ready = False

    def is_ready(self):
        return self.ready

    def put_event(self, event):
        pass

    def get_event(self):
        return None

    def qsize(self):
        pass

    def clean(self):
        pass


class RedisQueue(BaseQueue):
    def __init__(self, queue_name, host, port, password, max_connections):
        super(RedisQueue, self).__init__()
        self.queue_name = queue_name
        self.pool = redis.ConnectionPool(host=host, port=port, password=password, max_connections=max_connections,
                                         db=REDIS_QUEUE_DB_INDEX)
        if self.is_ready():
            stat_logger.info('init redis queue')
        else:
            stat_logger.error('init redis queue error!')
            raise Exception('init redis queue error!')

    def get_conn(self):
        return redis.Redis(connection_pool=self.pool, decode_responses=True)

    def get_event(self):
        try:
            conn = self.get_conn()
            content = conn.brpop([self.queue_name])
            event = self.parse_event(content[1])
            stat_logger.info('get event from redis queue: {}'.format(event))
            return event
        except Exception as e:
            stat_logger.error('get event from redis queue failed')
            stat_logger.exception(e)
            return None

    def is_ready(self):
        self.ready = self.test_connection()
        return self.ready

    def put_event(self, event):
        try:
            conn = self.get_conn()
            ret = conn.lpush(self.queue_name, json.dumps(event))
            stat_logger.info('put event into redis queue {}: {}'.format('successfully' if ret else 'failed', event))
        except Exception as e:
            stat_logger.error('put event into redis queue failed')
            stat_logger.exception(e)
            raise e

    def del_event(self, event):
        try:
            conn = self.get_conn()
            ret = conn.lrem(self.queue_name, 1, json.dumps(event))
            stat_logger.info('delete event from redis queue {}: {}'.format('successfully' if ret else 'failed', event))
            if not ret:
                raise Exception('job not in redis queue')
        except Exception as e:
            stat_logger.info('delete event from redis queue failed:{}'.format(str(e)))
            raise Exception('delete event from redis queue failed')

    def parse_event(self, content):
        return json.loads(content.decode())

    def test_connection(self):
        try:
            conn = self.get_conn()
            if (conn.echo("cccccc")):
                return True
            else:
                return False
        except BaseException:
            return False

    def qsize(self):
        conn = self.get_conn()
        return conn.llen(self.queue_name)


class MysqlQueue(BaseQueue):
    def __init__(self):
        super(MysqlQueue, self).__init__()
        self.ready = True
        self.mutex = threading.Lock()
        self.not_empty = threading.Condition(self.mutex)
        self.not_full = threading.Condition(self.mutex)
        self.maxsize = 0
        stat_logger.info('init queue')

    @staticmethod
    def lock(db, lock_name, timeout):
        sql = "SELECT GET_LOCK('%s', %s)" % (lock_name, timeout)
        stat_logger.info('lock mysql, lockname {}'.format(lock_name))
        cursor = db.execute_sql(sql)
        ret = cursor.fetchone()
        if ret[0] == 0:
            raise Exception('mysql lock {} is already used'.format(lock_name))
        elif ret[0] == 1:
            return True
        else:
            raise Exception('mysql lock {} error occurred!')

    @staticmethod
    def unlock(db, lock_name):
        sql = "SELECT RELEASE_LOCK('%s')" % (lock_name)
        stat_logger.info('unlock mysql, lockname {}'.format(lock_name))
        cursor = db.execute_sql(sql)
        ret = cursor.fetchone()
        if ret[0] == 0:
            raise Exception('mysql lock {} is not released'.format(lock_name))
        elif ret[0] == 1:
            return True
        else:
            raise Exception('mysql lock {} did not exist.'.format(lock_name))

    def put_event(self, event):
        try:
            self.put(event)
            stat_logger.info('put event into queue successfully: {}'.format(event))
        except Exception as e:
            stat_logger.error('put event into queue failed')
            stat_logger.exception(e)
            raise e

    def put(self, item, block=True, timeout=None):
        with self.not_full:
            with DB.connection_context():
                error = None
                MysqlQueue.lock(DB, 'fate_flow_job_queue', 10)
                try:
                    self.update_event(item=item)
                except Exception as e:
                    error =e
                MysqlQueue.unlock(DB, 'fate_flow_job_queue')
                if error:
                    raise Exception(e)
            self.not_empty.notify()

    def get_event(self):
        try:
            event = self.get(block=True)
            stat_logger.info('get event from queue successfully: {}'.format(event))
            return event
        except Exception as e:
            stat_logger.error('get event from queue failed')
            stat_logger.exception(e)
            return None

    def get(self, block=True, timeout=None):
        with self.not_empty:
            if not block:
                if not self.query_events():
                    raise Exception
            elif timeout is None:
                while not self.query_events():
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time() + timeout
                while not self.query_events():
                    remaining = endtime - time()
                    if remaining <= 0.0:
                        raise Exception
                    self.not_empty.wait(remaining)
            with DB.connection_context():
                error = None
                MysqlQueue.lock(DB, 'fate_flow_job_queue', 10)
                try:
                    item = Queue.select().where(Queue.f_is_waiting == 1)[0]
                    if item:
                        self.update_event(item.f_job_id)
                except Exception as e:
                    error = e
                MysqlQueue.unlock(DB, 'fate_flow_job_queue')
                if error:
                    raise Exception(e)
                self.not_full.notify()
                return json.loads(item.f_event)

    def del_event(self, event):
        ret = self.dell(event)
        if not ret:
            raise Exception('delete event failed, {} not in MysqlQueue'.format(event))
        else:
            stat_logger.info('delete event from  queue success: {}'.format(event))

    def query_events(self):
        with DB.connection_context():
            events = Queue.select().where(Queue.f_is_waiting == 1)
            return [event for event in events]

    def update_event(self, job_id=None, item=None):
        if job_id:
            event = Queue.select().where(Queue.f_job_id == job_id)[0]
            event.f_is_waiting = 0
        else:
            event = Queue()
            event.f_job_id = item.get('job_id')
            event.f_event = json.dumps(item)
        event.save()

    def dell(self, item):
        with self.not_empty:
            with DB.connection_context():
                MysqlQueue.lock(DB, 'fate_flow_job_queue', 10)
                del_status = True
                try:
                    job_id = item.get('job_id')
                    event = Queue.select().where(Queue.f_job_id == job_id)[0]
                    if event.f_is_waiting != 1:
                        del_status = False
                    event.f_is_waiting = 2
                    event.save()
                except Exception as e:
                    stat_logger.exception(e)
                    del_status = False
                MysqlQueue.unlock(DB, 'fate_flow_job_queue')
            self.not_full.notify()
        return del_status


class InProcessQueue(BaseQueue):
    def __init__(self):
        super(InProcessQueue, self).__init__()
        self.queue = queue.Queue()
        self.ready = True
        stat_logger.info('init in-process queue')

    def put_event(self, event):
        try:
            self.queue.put(event)
            stat_logger.info('put event into in-process queue successfully: {}'.format(event))
        except Exception as e:
            stat_logger.error('put event into in-process queue failed')
            stat_logger.exception(e)
            raise e

    def get_event(self):
        try:
            event = self.queue.get(block=True)
            stat_logger.info('get event from in-process queue successfully: {}'.format(event))
            return event
        except Exception as e:
            stat_logger.error('get event from in-process queue failed')
            stat_logger.exception(e)
            return None

    def qsize(self):
        return self.queue.qsize()


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

    def put_event(self, event):
        try:
            self.put(event)
            stat_logger.info('put event into in-process queue successfully: {}'.format(event))
        except Exception as e:
            stat_logger.error('put event into in-process queue failed')
            stat_logger.exception(e)
            raise e

    def get_event(self):
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

    def qsize(self):
        return len(self.queue)


def init_job_queue():
    if RuntimeConfig.WORK_MODE == WorkMode.STANDALONE:
        job_queue = ListQueue()
        RuntimeConfig.init_config(JOB_QUEUE=job_queue)
    elif RuntimeConfig.WORK_MODE == WorkMode.CLUSTER:
        # job_queue = RedisQueue(queue_name='fate_flow_job_queue', host=REDIS['host'], port=REDIS['port'],
        #                        password=REDIS['password'], max_connections=REDIS['max_connections'])
        job_queue = MysqlQueue()
        RuntimeConfig.init_config(JOB_QUEUE=job_queue)
    else:
        raise Exception('init queue failed.')

