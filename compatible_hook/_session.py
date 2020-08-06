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
from typing import Iterable

from fate_arch import session
from fate_arch.abc import CTableABC
from fate_arch.common import WorkMode, Backend
from fate_arch.common.log import getLogger

LOGGER = getLogger()


def init(job_id=None,
         mode: typing.Union[int, WorkMode] = WorkMode.STANDALONE,
         backend: typing.Union[int, Backend] = Backend.EGGROLL,
         options: dict = None,
         **kwargs):
    if kwargs:
        LOGGER.warning(f"{kwargs} not used, check!")

    if session.has_default():
        return session.default()
    return session.init(job_id, mode, backend, options)


def table(name, namespace, **kwargs) -> CTableABC:
    return session.default().computing.load(name=name, namespace=namespace, **kwargs)


def parallelize(data: Iterable, partition, include_key=False, **kwargs) -> CTableABC:
    return session.default().computing.parallelize(data=data, partition=partition, include_key=include_key, **kwargs)


def cleanup(name, namespace, *args, **kwargs):
    if len(args) > 0 or len(kwargs) > 0:
        LOGGER.warning(f"some args removed, please check! {args}, {kwargs}")
    return session.default().computing.cleanup(name=name, namespace=namespace)


def get_session_id():
    return session.default().computing.session_id


def get_data_table(name, namespace):
    LOGGER.warning(f"don't use this, use table directly")
    return session.default().computing.load(name=name, namespace=namespace)


def clean_tables(namespace, regex_string='*'):
    session.default().computing.cleanup(namespace=namespace, name=regex_string)


def stop():
    session.default().computing.stop()


def kill():
    session.default().computing.kill()


def exit():
    session.exit_session()
