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

from fate_arch.common.log import getLogger
import inspect
from functools import wraps
from fate_arch.abc import CTableABC

profile_logger = getLogger("PROFILING")
_PROFILE_LOG_ENABLED = True


class _ComputingTimer(object):
    _STATS = {}

    def __init__(self, name: str):
        self._start = time.time()
        self._name = name
        self._elapse = None

        if name not in self._STATS:
            self._STATS[name] = [0, 0.0]

    def done(self):
        self._elapse = time.time() - self._start
        self._STATS[self._name][1] += self._elapse
        self._STATS[self._name][0] += 1

    def elapse(self):
        return self._elapse

    @classmethod
    def computing_statistics_str(cls):
        return pprint.pformat(cls._STATS)


class _FederationTimer(object):
    _STATS = {}

    @classmethod
    def federation_statistics_str(cls):
        return pprint.pformat(cls._STATS)


class _FederationRemoteTimer(_FederationTimer):
    def __init__(self, name, tag, local, parties):
        self._name = name
        self._tag = tag
        self._local_party = local
        self._parties = parties
        self._start_time = time.time()
        self._end_time = None

        if name not in self._STATS:
            self._STATS[name] = {"get": 0, "remote": 0}
        self._STATS[name]["remote"] += 1

    def done(self, federation):
        self._end_time = time.time()
        profile_logger.info(f"[federation.remote@{self._local_party}->{self._parties}]"
                            f"done: name={self._name}, tag={self._tag}")

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
    def __init__(self, name, tag, local, parties):
        self._name = name
        self._tag = tag
        self._local_party = local
        self._parties = parties
        self._start_time = time.time()
        self._end_time = None

        if name not in self._STATS:
            self._STATS[name] = {"get": 0, "remote": 0}
        self._STATS[name]["get"] += 1

    def done(self, federation):
        self._end_time = time.time()
        profile_logger.info(f"[federation.get@{self._local_party}<-{self._parties}]"
                            f"done: name={self._name}, tag={self._tag}")

        if is_profile_remote_enable():
            remote_meta = federation.get(name=self._name, tag=profile_remote_tag(self._tag), parties=self._parties,
                                         gc=None)
            for party, meta in zip(self._parties, remote_meta):
                profile_logger.info(f"[federation.meta{self._local_party}<-{party}]"
                                    f"name={self._name}, tag = {self._tag}, meta={meta}")

    @property
    def elapse(self):
        return self._end_time - self._start_time


def federation_remote_timer(name, tag, local, parties):
    profile_logger.info(f"[federation.remote@{local}->{parties}]start: name={name}, tag={tag}")
    return _FederationRemoteTimer(name, tag, local, parties)


def federation_get_timer(name, tag, local, parties):
    profile_logger.info(f"[federation.get@{local}<-{parties}]start: name={name}, tag={tag}")
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
        call_stack_strings = _call_stack_strings()
        if _PROFILE_LOG_ENABLED:
            profile_logger.info(f"[computing.{func.__name__}]start, "
                                f"func: {func_string}, "
                                f"call_stack: {call_stack_strings}")
        timer = _ComputingTimer(func.__name__)
        rtn = func(*args, **kwargs)
        timer.done()
        if _PROFILE_LOG_ENABLED:
            profile_logger.info(f"[computing.{func.__name__}]done, func: {func_string}->{_pretty_table_str(rtn)}, "
                                f"elapse={timer.elapse()}, "
                                f"call_stack: {call_stack_strings}")
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
