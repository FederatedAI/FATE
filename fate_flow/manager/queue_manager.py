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
import queue

import redis

from fate_flow.entity.runtime_config import RuntimeConfig
from fate_flow.entity.constant_config import WorkMode
from fate_flow.settings import REDIS, REDIS_QUEUE_DB_INDEX, stat_logger


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


class RedisQueue(BaseQueue):
    def __init__(self, queue_name, host, port, password, max_connections):
        super(RedisQueue, self).__init__()
        self.queue_name = queue_name
        self.pool = redis.ConnectionPool(host=host, port=port, password=password, max_connections=max_connections,
                                         db=REDIS_QUEUE_DB_INDEX)
        stat_logger.info('init redis queue')

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
            stat_logger.exception(e)
            stat_logger.error('get event from redis queue failed')
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
            stat_logger.exception(e)
            stat_logger.error('put event into redis queue failed')

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
            stat_logger.exception(e)
            stat_logger.error('put event into in-process queue failed')

    def get_event(self):
        try:
            event = self.queue.get(block=True)
            stat_logger.info('get event from in-process queue successfully: {}'.format(event))
            return event
        except Exception as e:
            stat_logger.exception(e)
            stat_logger.error('get event from in-process queue failed')
            return None

    def qsize(self):
        return self.queue.qsize()


def init_job_queue():
    if RuntimeConfig.WORK_MODE == WorkMode.STANDALONE:
        job_queue = InProcessQueue()
        RuntimeConfig.init_config(JOB_QUEUE=job_queue)
    elif RuntimeConfig.WORK_MODE == WorkMode.CLUSTER:
        job_queue = RedisQueue(queue_name='fate_flow_job_queue', host=REDIS['host'], port=REDIS['port'],
                               password=REDIS['password'], max_connections=REDIS['max_connections'])
        RuntimeConfig.init_config(JOB_QUEUE=job_queue)
    else:
        raise Exception('init queue failed.')
