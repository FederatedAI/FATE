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


import concurrent.futures
import typing

from fate_arch.common.log import getLogger

LOGGER = getLogger()

_remote_futures = set()


def _clear_callback(future):
    LOGGER.debug("future `{future}` done, remove")
    _remote_futures.remove(future)


def add_remote_futures(fs: typing.List[concurrent.futures.Future]):
    for f in fs:
        f.add_done_callback(_clear_callback)
        _remote_futures.add(f)


def wait_all_remote_done(timeout=None):
    concurrent.futures.wait(
        _remote_futures, timeout=timeout, return_when=concurrent.futures.ALL_COMPLETED
    )
