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
from logging import getLogger

import typing

from fate_arch._interface import GC

LOGGER = getLogger()


class IterationGC(GC):
    def __init__(self, capacity=2):
        self._ashcan: deque[typing.List] = deque()
        self._last_tag: typing.Optional[str] = None
        self._capacity = capacity
        self._enable = True

    def add_gc_func(self, tag: str, func: typing.Callable[[], typing.NoReturn]):
        if self._last_tag == tag:
            self._ashcan[-1].append(func)
        else:
            self._ashcan.append([func])
            self._last_tag = tag

    def disable(self):
        self._enable = False

    def set_capacity(self, capacity):
        self._capacity = capacity

    def gc(self):
        if len(self._ashcan) <= self._capacity:
            return
        self._save_gc_call(self._ashcan.pop())

    def clean(self):
        while self._ashcan:
            self._save_gc_call(self._ashcan.pop())

    @staticmethod
    def _save_gc_call(funcs):
        for func in funcs:
            try:
                func()
            except Exception as e:
                LOGGER.debug(f"[CLEAN]this could be ignore {e}")
