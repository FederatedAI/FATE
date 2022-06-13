#
#  Copyright 2022 The FATE Authors. All Rights Reserved.
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

import random
import time

from fate_arch.common.log import getLogger

LOGGER = getLogger()


def nretry1(func):
    """retry
    """

    def wrapper(self, *args, **kwargs):
        """wrapper
        """
        cur_retry = 0
        exception = None
        ret = None
        max_retry_cnt = 3
        long_retry_cnt = 2

        while cur_retry < max_retry_cnt:
            LOGGER.debug(
                f'cur_retry={cur_retry}')
            try:
                ret = func(self, *args, **kwargs)
                exception = None
                break
            except Exception as e:
                if cur_retry < max_retry_cnt - long_retry_cnt:
                    retry_interval = round(min(2 * cur_retry, 20) + random.random() * 10, 3)
                else:
                    retry_interval = round(300 + random.random() * 10, 3)
                LOGGER.debug(
                    f"max_retry_cnt={max_retry_cnt}, cur_retry={cur_retry}, retry_interval={retry_interval}",
                    exc_info=e)
                time.sleep(retry_interval)
                exception = e
            finally:
                cur_retry += 1
        if exception is not None:
            LOGGER.exception(
                f"failed. cur_retry={cur_retry}",
                exc_info=exception)
            raise exception

        return ret

    return wrapper


def nretry(func):
    """retry connection
    """

    def wrapper(self, *args, **kwargs):
        """wrapper
        """
        res = None
        exception = None
        for ntry in range(10):
            try:
                res = func(self, *args, **kwargs)
                exception = None
                break
            except Exception as e:
                LOGGER.error("function %s error" % func.__name__, exc_info=True)
                exception = e
                time.sleep(0.1)

        if exception is not None:
            LOGGER.debug(
                f"failed",
                exc_info=exception)
            raise exception

        return res

    return wrapper
