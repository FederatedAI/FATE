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

import time

from fate_arch.common.log import getLogger
import inspect
from functools import wraps
from fate_arch.abc import CTableABC

profile_logger = getLogger("PROFILING")


def log_elapsed(func):
    func_name = func.__name__

    @wraps(func)
    def _fn(*args, **kwargs):
        t = time.time()
        name = f"{func_name}#{kwargs['func_tag']}" if 'func_tag' in kwargs else func_name
        rtn = func(*args, **kwargs)
        frame = inspect.getouterframes(inspect.currentframe(), 2)
        profile_logger.debug(f"{frame[1].filename.split('/')[-1]}:{frame[1].lineno} call {name}, takes {time.time() - t}s")
        try:
            profile_logger.debug("call %s partitions %d" % (name, rtn.partitions))
        except:
            profile_logger.debug("")
        return rtn

    return _fn


def _pretty_table_str(v):
    if isinstance(v, CTableABC):
        return f"<Table: partition={v.partitions}>"
    else:
        return f"<{type(v).__name__}>"


def _pretty_func_apply_string(func, *args, **kwargs):
    pretty_args = []
    for k, v in inspect.signature(func).bind(*args, **kwargs).arguments.items():
        pretty_args.append(f"{k}={_pretty_table_str(v)}")
    return f"{func.__name__}({', '.join(pretty_args)})"


def computing_profile(func):
    @wraps(func)
    def _fn(*args, **kwargs):
        func_apply_string = _pretty_func_apply_string(func, *args, **kwargs)
        profile_logger.info(f"[{func_apply_string}]start")
        t = time.time()
        rtn = func(*args, **kwargs)
        profile_logger.info(f"[{func_apply_string}->{_pretty_table_str(rtn)}]done, takes {time.time() - t}")
        return rtn

    return _fn


__META_REMOTE_ENABLE = True


def enable_meta_remote():
    global __META_REMOTE_ENABLE
    __META_REMOTE_ENABLE = True


def is_meta_remote_enable():
    return __META_REMOTE_ENABLE


def remote_meta_tag(tag):
    return f"<remote_meta>_{tag}"
