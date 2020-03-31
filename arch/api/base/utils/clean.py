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
from collections import deque

from arch.api.utils.log_utils import getLogger

LOGGER = getLogger()


class Rubbish(object):
    """
    a collection collects all tables / objects in federation tagged by `tag`.
    """

    def __init__(self, name, tag):
        self._name = name
        self._tag = tag
        self._tables = []
        self._kv = {}

    @property
    def tag(self):
        return self._tag

    def add_table(self, table):
        self._tables.append(table)

    # noinspection PyProtectedMember
    def add_obj(self, table, key):
        if (table._name, table._namespace) not in self._kv:
            self._kv[(table._name, table._namespace)] = (table, [key])
        else:
            self._kv[(table._name, table._namespace)][1].append(key)

    def merge(self, rubbish: 'Rubbish'):
        self._tables.extend(rubbish._tables)
        for tk, (t, k) in rubbish._kv.items():
            if tk in self._kv:
                self._kv[tk][1].extend(k)
            else:
                self._kv[tk] = (t, k)
        # # warm: this is necessary to prevent premature clean work invoked by `__del__` in `rubbish`
        # rubbish.empty()
        return self

    def empty(self):
        self._tables = []
        self._kv = {}

    # noinspection PyBroadException
    def clean(self):
        if self._tables or self._kv:
            LOGGER.debug(f"[CLEAN] {self._name} cleaning rubbishes tagged by {self._tag}")
        for table in self._tables:
            try:
                LOGGER.debug(f"[CLEAN] try destroy table {table}")
                table.destroy()
            except:
                pass

        for _, (table, keys) in self._kv.items():
            for key in keys:
                try:
                    LOGGER.debug(f"[CLEAN] try delete object with key={key} from table={table}")
                    table.delete(key)
                except:
                    pass

    # def __del__(self):  # this is error prone, please call `clean` explicit
    #     self.clean()


class Cleaner(object):
    def __init__(self):
        self._ashcan: deque[Rubbish] = deque()

    def push(self, rubbish):
        """
        append `rubbish`
        :param rubbish: a rubbish instance
        :return:
        """
        if self.is_latest_tag(rubbish.tag):
            self._ashcan[-1].merge(rubbish)
        else:
            self._ashcan.append(rubbish)
        return self

    def is_latest_tag(self, tag):
        return len(self._ashcan) > 0 and self._ashcan[-1].tag == tag

    def keep_latest_n(self, n):
        while len(self._ashcan) > n:
            self._ashcan.popleft().clean()  # clean explicit

    def clean_all(self):
        while len(self._ashcan) > 0:
            self._ashcan.popleft().clean()
