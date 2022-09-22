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

import typing
from collections import deque

from fate_arch.abc import GarbageCollectionABC
from fate_arch.common.log import getLogger

LOGGER = getLogger()


class IterationGC(GarbageCollectionABC):
    def __init__(self, capacity=2):
        self._ashcan: deque[typing.List[typing.Tuple[typing.Any, str, dict]]] = deque()
        self._last_tag: typing.Optional[str] = None
        self._capacity = capacity
        self._enable = True

    def add_gc_action(self, tag: str, obj, method, args_dict):
        if self._last_tag == tag:
            self._ashcan[-1].append((obj, method, args_dict))
        else:
            self._ashcan.append([(obj, method, args_dict)])
            self._last_tag = tag

    def disable(self):
        self._enable = False

    def set_capacity(self, capacity):
        self._capacity = capacity

    def gc(self):
        if not self._enable:
            return
        if len(self._ashcan) <= self._capacity:
            return
        self._safe_gc_call(self._ashcan.popleft())

    def clean(self):
        while self._ashcan:
            self._safe_gc_call(self._ashcan.pop())

    @staticmethod
    def _safe_gc_call(actions: typing.List[typing.Tuple[typing.Any, str, dict]]):
        for obj, method, args_dict in actions:
            try:
                LOGGER.debug(f"[CLEAN]deleting {obj}, {method}, {args_dict}")
                getattr(obj, method)(**args_dict)
            except Exception as e:
                LOGGER.debug(f"[CLEAN]this could be ignore {e}")
