#
#  Copyright 2019 The Eggroll Authors. All Rights Reserved.
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

from cachetools import LRUCache, TTLCache
import time


class EvictLRUCache(LRUCache):

    def __init__(self, maxsize, getsizeof=None, evict=None):
        LRUCache.__init__(self, maxsize, getsizeof)
        self.__evict = evict

    '''
    eviction callback
    '''

    def popitem(self):
        key, val = LRUCache.popitem(self)
        evict = self.__evict
        if evict:
            evict(key, val)
        return key, val


class EvictTTLCache(TTLCache):

    def __init__(self, maxsize, ttl, timer=time.time, getsizeof=None, evict=None):
        TTLCache.__init__(self, maxsize, ttl, timer, getsizeof)
        self.__evict = evict

    '''
    eviction callback
    '''
    def popitem(self):
        key, val = TTLCache.popitem(self)
        evict = self.__evict
        if evict:
            evict(key, val)
        return key, val
