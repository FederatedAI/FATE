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
import os
from arch.api.utils import log_utils
from fate_flow.settings import REDIS, REDIS_QUEUE_DB_INDEX
import redis

LOGGER = log_utils.getLogger()


def singleton(cls, *args, **kw):
    instances = {}

    def _singleton():
        key = str(cls) + str(os.getpid())
        if key not in instances:
            instances[key] = cls(*args, **kw)
        return instances[key]

    return _singleton


@singleton
class RedisAdaptor(object):
    def __init__(self):
        config = REDIS.copy()
        self.pool = redis.ConnectionPool(host=config['host'], port=config['port'], password=config['password'], \
             max_connections=config['max_connections'], db=REDIS_QUEUE_DB_INDEX)
        LOGGER.info('init redis connection pool.')

    def get_conn(self):
        return redis.Redis(connection_pool=self.pool, decode_responses=True)

    def get(self, key):
        try:
            conn = self.get_conn()
            value = conn.get(key)
            if value:
                LOGGER.info('get from redis, {}:{}'.format(key, value))
            else:
                LOGGER.info('get from redis return nil, key={}'.format(key))
            return value
        except Exception as e:
            LOGGER.exception(e)
            LOGGER.error('get from redis failed')
            return None

    def setex(self, key, value, expire_seconds=10800):
        try:
            conn = self.get_conn()
            conn.setex(key, expire_seconds, value)
            LOGGER.info('set {}:{} {} into redis.'.format(key, value, expire_seconds))
        except Exception as e:
            LOGGER.exception(e)
            LOGGER.info('set {}:{} {} into redis failed.'.format(key, value, expire_seconds))

    def delete(self, *key):
        try:
            conn = self.get_conn()
            conn.delete(*key)
            LOGGER.info('del {} from redis.'.format(*key))
        except Exception as e:
            LOGGER.exception(e)
            LOGGER.info('del {} from redis failed.'.format(*key))

