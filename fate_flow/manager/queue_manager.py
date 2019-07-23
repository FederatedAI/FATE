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
import redis
import json
from fate_flow.settings import REDIS, REDIS_QUEUE_DB_INDEX


class BaseQueue:
    RESULT_KEY = "resultKey"

    def __init__(self):
        self.ready = False

    def get_event(self):
        return None

    def is_ready(self):
        return self.ready

    def set_result(self, params):
        pass

    def qsize(self):
        pass


class RedisQueue(BaseQueue):
    def __init__(self, queue_name, host, port, password, max_connections):
        self.queue_name = queue_name
        self.pool = redis.ConnectionPool(host=host, port=port, password=password, max_connections=max_connections, db=REDIS_QUEUE_DB_INDEX)

    def get_conn(self):
        return redis.Redis(connection_pool=self.pool, decode_responses=True)

    def get_event(self):
        conn = self.get_conn()
        content = conn.brpop([self.queue_name])
        return self.parse_event(content[1])

    def is_ready(self):
        self.ready = self.test_connection()
        return self.ready

    def put_event(self, event):
        conn = self.get_conn()
        ret = conn.lpush(self.queue_name, json.dumps(event))

    def parse_event(self, content):
        return json.loads(content.decode())

    def set_result(self, resultKey, result):
        if (self.ready):
            conn = self.get_conn()
            return conn.set(resultKey, result)
        else:
            pass

    def get_result(self, resultKey):
        if (self.ready):
            conn = self.get_conn()
            print("getResult===========")
            return conn.get(resultKey)
        else:
            pass

    def format_event(self, event):
        return json.dump(event)

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


JOB_QUEUE = RedisQueue(queue_name='fate_flow_job_queue', host=REDIS['host'], port=REDIS['port'], password=REDIS['password'], max_connections=REDIS['max_connections'])
