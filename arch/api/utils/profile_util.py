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

from arch.api.utils import log_utils
import inspect

LOGGER = log_utils.getLogger("PROFILING")


def log_elapsed(func):
    func_name = func.__name__

    def _fn(*args, **kwargs):
        t = time.time()
        name = f"{func_name}#{kwargs['func_tag']}" if 'func_tag' in kwargs else func_name
        rtn = func(*args, **kwargs)
        frame = inspect.getouterframes(inspect.currentframe(), 2)
        LOGGER.debug(f"{frame[1].filename.split('/')[-1]}:{frame[1].lineno} call {name}, takes {time.time() - t}s")
        return rtn
    return _fn
