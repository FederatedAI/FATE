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
import pprint
import time
import typing
import uuid

from fate_arch.common.log import getLogger
import inspect
from functools import wraps
from fate_arch.abc import CTableABC

profile_logger = getLogger("PROFILING")
_PROFILE_LOG_ENABLED = True


class _ComputingTimerItem(object):
    def __init__(self):
        self.count = 0
        self.total_time = 0.0
        self.max_time = 0.0
        self.max_time_uuid = None

    def add(self, elapse_time, computing_uuid):
        self.count += 1
        self.total_time += elapse_time
        if elapse_time > self.max_time:
            self.max_time = elapse_time
            self.max_time_uuid = computing_uuid

    def get_statistic(self):
        return [self.count, self.total_time, self.total_time / self.count, self.max_time, self.max_time_uuid]

    def __str__(self):
        return f"count={self.count}, total_time={self.total_time}, mean_time={self.total_time / self.count}," \
               f" max_time={self.max_time}, top_cost_computing_uuid={self.max_time_uuid}"

    def __repr__(self):
        return self.__str__()


class _ComputingTimer(object):
    _STATS: typing.MutableMapping[str, _ComputingTimerItem] = {}

    def __init__(self, name: str, call_stack):
        self._start = time.time()
        self._name = name
        self._elapse = None
        self.uuid = uuid.uuid1()

        if name not in self._STATS:
            self._STATS[name] = _ComputingTimerItem()

        profile_logger.debug(f"[computing.{self._name}]uuid={self.uuid}, call_stack={call_stack}")

    def done(self):
        self._elapse = time.time() - self._start
        self._STATS[self._name].add(self._elapse, self.uuid)

    def elapse(self):
        return self._elapse

    @classmethod
    def computing_statistics_str(cls):
        try:
            # noinspection PyPackageRequirements
            import prettytable
        except ImportError:
            return pprint.pformat(cls._STATS)
        else:
            head = ["name", "count", "total_time", "mean_time", "max_time", "most_cost_computing_uuid"]
            pretty_table = prettytable.PrettyTable(head)
            pretty_table.hrules = prettytable.ALL
            pretty_table.max_width["name"] = 25
            for name, timer in cls._STATS.items():
                pretty_table.add_row([name, *timer.get_statistic()])
            return pretty_table.get_string()


class _FederationTimerItem(object):
    def __init__(self):
        self.get_count = 0
        self.remote_count = 0
        self.get_time = 0.0
        self.remote_time = 0.0

    @property
    def get_mean_time(self):
        if self.get_count > 0:
            return self.get_time / self.get_count
        else:
            return 0.0

    @property
    def remote_mean_time(self):
        if self.remote_count > 0:
            return self.remote_time / self.remote_count
        else:
            return 0.0

    def get_statistic(self):
        return [self.get_count, self.remote_count, self.get_time, self.remote_time, self.get_mean_time,
                self.remote_mean_time]

    def __str__(self):
        return f"get_count={self.get_count}, remote_count={self.remote_count}, " \
               f"get_time={self.get_time}, remote_time={self.remote_time}" \
               f"get_mean_time={self.get_time / self.get_count}, " \
               f"remote_mean_time={self.remote_time / self.remote_count}"

    def __repr__(self):
        return self.__str__()


class _FederationTimer(object):
    _STATS: typing.MutableMapping[str, _FederationTimerItem] = {}

    @classmethod
    def federation_statistics_str(cls):
        try:
            # noinspection PyPackageRequirements
            import prettytable
        except ImportError:
            return pprint.pformat(cls._STATS)
        else:
            head = ["name", "get_count", "remote_count", "get_time", "remote_time", "mean_get_time", "mean_remote_time"]
            pretty_table = prettytable.PrettyTable(head)
            pretty_table.hrules = prettytable.ALL
            pretty_table.max_width["name"] = 25
            for name, timer in cls._STATS.items():
                pretty_table.add_row([name, *timer.get_statistic()])
            return pretty_table.get_string()


class _FederationRemoteTimer(_FederationTimer):
    def __init__(self, name, tag, local, parties):
        self._name = name
        self._tag = tag
        self._local_party = local
        self._parties = parties
        self._start_time = time.time()
        self._end_time = None

        if name not in self._STATS:
            self._STATS[name] = _FederationTimerItem()
        self._STATS[name].remote_count += 1

    def done(self, federation):
        self._end_time = time.time()

        profile_logger.debug(f"[federation.remote@{self._local_party}->{self._parties}]"
                             f"done: name={self._name}, tag={self._tag}")

        if is_profile_remote_enable():
            federation.remote(v={"start_time": self._start_time, "end_time": self._end_time},
                              name=self._name,
                              tag=profile_remote_tag(self._tag),
                              parties=self._parties,
                              gc=None)
        self._STATS[self._name].remote_time += self.elapse

    @property
    def elapse(self):
        return self._end_time - self._start_time


class _FederationGetTimer(_FederationTimer):
    def __init__(self, name, tag, local, parties):
        self._name = name
        self._tag = tag
        self._local_party = local
        self._parties = parties
        self._start_time = time.time()
        self._end_time = None

        if name not in self._STATS:
            self._STATS[name] = _FederationTimerItem()
        self._STATS[name].get_count += 1

    def done(self, federation):
        self._end_time = time.time()
        profile_logger.debug(f"[federation.get@{self._local_party}<-{self._parties}]"
                             f"done: name={self._name}, tag={self._tag}")

        if is_profile_remote_enable():
            remote_meta = federation.get(name=self._name, tag=profile_remote_tag(self._tag), parties=self._parties,
                                         gc=None)
            for party, meta in zip(self._parties, remote_meta):
                profile_logger.debug(f"[federation.meta{self._local_party}<-{party}]"
                                     f"name={self._name}, tag = {self._tag}, meta={meta}")
        self._STATS[self._name].get_time += self.elapse

    @property
    def elapse(self):
        return self._end_time - self._start_time


def federation_remote_timer(name, tag, local, parties):
    profile_logger.debug(f"[federation.remote@{local}->{parties}]start: name={name}, tag={tag}")
    return _FederationRemoteTimer(name, tag, local, parties)


def federation_get_timer(name, tag, local, parties):
    profile_logger.debug(f"[federation.get@{local}<-{parties}]start: name={name}, tag={tag}")
    return _FederationGetTimer(name, tag, local, parties)


def profile_ends():
    profile_logger.info(f"computing_statistics:\n{_ComputingTimer.computing_statistics_str()}")
    profile_logger.info(f"federation_statistics:\n{_FederationTimer.federation_statistics_str()}")
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
    frames = inspect.getouterframes(inspect.currentframe(), 10)[2:]
    for frame in frames:
        call_stack_strings.append(f"{frame.filename.split('/')[-1]}:{frame.function}:{frame.lineno}")
    return call_stack_strings


def computing_profile(func):
    @wraps(func)
    def _fn(*args, **kwargs):
        func_string = _func_annotated_string(func, *args, **kwargs)
        timer = _ComputingTimer(func.__name__, _call_stack_strings())
        if _PROFILE_LOG_ENABLED:
            profile_logger.debug(f"[computing.{func.__name__}]start, func: {func_string}, uuid={timer.uuid}")
        rtn = func(*args, **kwargs)
        timer.done()
        if _PROFILE_LOG_ENABLED:
            profile_logger.debug(f"[computing.{func.__name__}]done, func: {func_string}->{_pretty_table_str(rtn)}, "
                                 f"elapse={timer.elapse()}, uuid={timer.uuid}")
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
