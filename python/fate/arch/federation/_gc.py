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

import logging
import typing
from collections import deque

from fate.arch.abc import GarbageCollector as GarbageCollectorInterface

LOGGER = logging.getLogger(__name__)


class GarbageCollector(GarbageCollectorInterface):
    def __init__(self) -> None:
        self._register = {}

    def register_clean_action(self, name: str, tag: str, obj, method: str, kwargs):
        """
        when clean action for (`name`, `tag`) triggered, do
        `getattr(obj, method)(**kwargs)`
        """
        self._register.setdefault(name, {})[tag] = (obj, method, kwargs)

    def clean(self, name: str, tag: str):
        if tag == "*":
            if name in self._register:
                for _, (obj, method, kwargs) in self._register[name].items():
                    self._safe_gc_call(obj, method, kwargs)
                del self._register[name]
        else:
            if name in self._register and tag in self._register[name]:
                obj, method, kwargs = self._register[name][tag]
                self._safe_gc_call(obj, method, kwargs)
                del self._register[name][tag]

    @classmethod
    def _safe_gc_call(cls, obj, method: str, kwargs: dict):
        try:
            LOGGER.debug(f"[CLEAN]deleting {obj}, {method}, {kwargs}")
            getattr(obj, method)(**kwargs)
        except Exception as e:
            LOGGER.debug(f"[CLEAN]this could be ignore {e}")


class IterationGC:
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
