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
import hashlib
import time
import typing

import beautifultable

from fate_arch.common.log import getLogger
import inspect
from functools import wraps
from fate_arch.abc import CTableABC

profile_logger = getLogger("PROFILING")
_PROFILE_LOG_ENABLED = True


class _TimerItem(object):
    def __init__(self):
        self.count = 0
        self.total_time = 0.0
        self.max_time = 0.0

    def union(self, other: '_TimerItem'):
        self.count += other.count
        self.total_time += other.total_time
        if self.max_time < other.max_time:
            self.max_time = other.max_time

    def add(self, elapse_time):
        self.count += 1
        self.total_time += elapse_time
        if elapse_time > self.max_time:
            self.max_time = elapse_time

    @property
    def mean(self):
        if self.count == 0:
            return 0.0
        return self.total_time / self.count

    def as_list(self):
        return [self.count, self.total_time, self.mean, self.max_time]

    def __str__(self):
        return f"n={self.count}, sum={self.total_time:.4f}, mean={self.mean:.4f}, max={self.max_time:.4f}"

    def __repr__(self):
        return self.__str__()


class _ComputingTimerItem(object):
    def __init__(self, function_name: str, function_stack):
        self.function_name = function_name
        self.function_stack = function_stack
        self.item = _TimerItem()


class _ComputingTimer(object):
    _STATS: typing.MutableMapping[str, _ComputingTimerItem] = {}

    def __init__(self, function_name: str, function_stack):
        self._start = time.time()

        function_stack = "\n".join(function_stack)
        self._hash = hashlib.blake2b(function_stack.encode('utf-8'), digest_size=5).hexdigest()

        if self._hash not in self._STATS:
            self._STATS[self._hash] = _ComputingTimerItem(function_name, function_stack)
            if _PROFILE_LOG_ENABLED:
                profile_logger.debug(f"[computing#{self._hash}]function_stack: {function_stack}")

        if _PROFILE_LOG_ENABLED:
            profile_logger.debug(f"[computing#{self._hash}]start")

    def done(self, function_string):
        elapse = time.time() - self._start
        self._STATS[self._hash].item.add(elapse)
        if _PROFILE_LOG_ENABLED:
            profile_logger.debug(f"[computing#{self._hash}]done, elapse: {elapse}, function: {function_string}")

    @classmethod
    def computing_statistics_table(cls):
        stack_table = beautifultable.BeautifulTable(110, precision=4)
        stack_table.set_style(beautifultable.STYLE_BOX_ROUNDED)
        stack_table.columns.header = ["function", "n", "sum(s)", "mean(s)", "max(s)", "stack_hash", "stack"]
        stack_table.columns.alignment["stack"] = beautifultable.ALIGN_LEFT
        stack_table.column_headers.alignment = beautifultable.ALIGN_CENTER
        stack_table.border.left = ''
        stack_table.border.right = ''
        stack_table.border.bottom = ''
        stack_table.border.top = ''

        function_table = beautifultable.BeautifulTable(110)
        function_table.set_style(beautifultable.STYLE_COMPACT)
        function_table.columns.header = ["function", "n", "sum(s)", "mean(s)", "max(s)"]

        aggregate = {}
        total = _TimerItem()
        for hash_id, timer in cls._STATS.items():
            stack_table.rows.append([timer.function_name, *timer.item.as_list(), hash_id, timer.function_stack])
            aggregate.setdefault(timer.function_name, _TimerItem()).union(timer.item)
            total.union(timer.item)

        for function_name, item in aggregate.items():
            function_table.rows.append([function_name, *item.as_list()])

        base_table = beautifultable.BeautifulTable(120)
        stack_table.rows.sort("sum(s)", reverse=True)
        base_table.rows.append(["stack", stack_table])
        function_table.rows.sort("sum(s)", reverse=True)
        base_table.rows.append(["function", function_table])
        base_table.rows.append(["total", total])

        return base_table.get_string()


class _FederationTimer(object):
    _GET_STATS: typing.MutableMapping[str, _TimerItem] = {}
    _REMOTE_STATS: typing.MutableMapping[str, _TimerItem] = {}

    @classmethod
    def federation_statistics_table(cls):
        total = _TimerItem()
        get_table = beautifultable.BeautifulTable(110)
        get_table.columns.header = ["name", "n", "sum(s)", "mean(s)", "max(s)"]
        for name, item in cls._GET_STATS.items():
            get_table.rows.append([name, *item.as_list()])
            total.union(item)
        get_table.rows.sort("sum(s)", reverse=True)
        get_table.border.left = ''
        get_table.border.right = ''
        get_table.border.bottom = ''
        get_table.border.top = ''
        remote_table = beautifultable.BeautifulTable(110)
        remote_table.columns.header = ["name", "n", "sum(s)", "mean(s)", "max(s)"]
        for name, item in cls._REMOTE_STATS.items():
            remote_table.rows.append([name, *item.as_list()])
            total.union(item)
        remote_table.rows.sort("sum(s)", reverse=True)
        remote_table.border.left = ''
        remote_table.border.right = ''
        remote_table.border.bottom = ''
        remote_table.border.top = ''

        base_table = beautifultable.BeautifulTable(120)
        base_table.rows.append(["get", get_table])
        base_table.rows.append(["remote", remote_table])
        base_table.rows.append(["total", total])
        return base_table.get_string()


class _FederationRemoteTimer(_FederationTimer):
    def __init__(self, name, full_name, tag, local, parties):
        self._name = name
        self._full_name = full_name
        self._tag = tag
        self._local_party = local
        self._parties = parties
        self._start_time = time.time()
        self._end_time = None

        if self._full_name not in self._REMOTE_STATS:
            self._REMOTE_STATS[self._full_name] = _TimerItem()

    def done(self, federation):
        self._end_time = time.time()
        self._REMOTE_STATS[self._full_name].add(self.elapse)
        profile_logger.debug(f"[federation.remote@{self._local_party}->{self._parties}]"
                             f"done: name={self._full_name}, tag={self._tag}")

        if is_profile_remote_enable():
            federation.remote(v={"start_time": self._start_time, "end_time": self._end_time},
                              name=self._name,
                              tag=profile_remote_tag(self._tag),
                              parties=self._parties,
                              gc=None)

    @property
    def elapse(self):
        return self._end_time - self._start_time


class _FederationGetTimer(_FederationTimer):
    def __init__(self, name, full_name, tag, local, parties):
        self._name = name
        self._full_name = full_name
        self._tag = tag
        self._local_party = local
        self._parties = parties
        self._start_time = time.time()
        self._end_time = None

        if self._full_name not in self._GET_STATS:
            self._GET_STATS[self._full_name] = _TimerItem()

    def done(self, federation):
        self._end_time = time.time()
        self._GET_STATS[self._full_name].add(self.elapse)
        profile_logger.debug(f"[federation.get@{self._local_party}<-{self._parties}]"
                             f"done: name={self._full_name}, tag={self._tag}")

        if is_profile_remote_enable():
            remote_meta = federation.get(name=self._name, tag=profile_remote_tag(self._tag), parties=self._parties,
                                         gc=None)
            for party, meta in zip(self._parties, remote_meta):
                profile_logger.debug(f"[federation.meta{self._local_party}<-{party}]"
                                     f"name={self._full_name}, tag = {self._tag}, meta={meta}")

    @property
    def elapse(self):
        return self._end_time - self._start_time


def federation_remote_timer(name, full_name, tag, local, parties):
    profile_logger.debug(f"[federation.remote@{local}->{parties}]start: name={name}, tag={tag}")
    return _FederationRemoteTimer(name, full_name, tag, local, parties)


def federation_get_timer(name, full_name, tag, local, parties):
    profile_logger.debug(f"[federation.get@{local}<-{parties}]start: name={name}, tag={tag}")
    return _FederationGetTimer(name, full_name, tag, local, parties)


def profile_ends():
    profile_logger.info(f"\n"
                        f"Computing:\n"
                        f"{_ComputingTimer.computing_statistics_table()}\n\n"
                        f"Federation:\n"
                        f"{_FederationTimer.federation_statistics_table()}\n")
    global _PROFILE_LOG_ENABLED
    _PROFILE_LOG_ENABLED = False


def _pretty_table_str(v):
    if isinstance(v, CTableABC):
        return f"Table(partition={v.partitions})"
    else:
        return f"{type(v).__name__}"


def _func_annotated_string(func, *args, **kwargs):
    pretty_args = []
    for k, v in inspect.signature(func).bind(*args, **kwargs).arguments.items():
        pretty_args.append(f"{k}: {_pretty_table_str(v)}")
    return f"{func.__name__}({', '.join(pretty_args)})"


def _call_stack_strings():
    call_stack_strings = []
    frames = inspect.getouterframes(inspect.currentframe(), 10)[2:-2]
    for frame in frames:
        call_stack_strings.append(f"[{frame.filename.split('/')[-1]}:{frame.lineno}]{frame.function}")
    return call_stack_strings


def computing_profile(func):
    @wraps(func)
    def _fn(*args, **kwargs):
        function_call_stack = _call_stack_strings()
        timer = _ComputingTimer(func.__name__, function_call_stack)
        rtn = func(*args, **kwargs)
        function_string = f"{_func_annotated_string(func, *args, **kwargs)} -> {_pretty_table_str(rtn)}"
        timer.done(function_string)
        return rtn

    return _fn


__META_REMOTE_ENABLE = True


def enable_profile_remote():
    global __META_REMOTE_ENABLE
    __META_REMOTE_ENABLE = True


def is_profile_remote_enable():
    return __META_REMOTE_ENABLE


def profile_remote_tag(tag):
    return f"<remote_profile>_{tag}"
