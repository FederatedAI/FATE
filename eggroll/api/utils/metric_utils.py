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
from eggroll.api.utils import log_utils

LOGGER = log_utils.getLogger()


def record_metrics(func):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kw):
        try:
            start = time.process_time()
            result = func(*args, **kw)
            result_status = 'success'
        except Exception:
            result_status = 'error'
            raise
        finally:
            end = time.process_time()
            LOGGER.debug('{}.{}: {} status: {}'.format(func.__module__, func.__name__, end - start, result_status))
        return result

    return wrapper
