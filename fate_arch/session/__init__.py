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

import uuid

from fate_arch.session._interface import SessionABC, TableABC
from fate_arch.common import WorkMode, Backend

_DEFAULT_SESSION: typing.Optional[SessionABC] = None

__all__ = ['create', 'default', 'has_default', 'is_table', 'TableABC']


def init(session_id=None,
         mode: typing.Union[int, WorkMode] = WorkMode.STANDALONE,
         backend: typing.Union[int, Backend] = Backend.EGGROLL,
         options: dict = None):
    if isinstance(mode, int):
        mode = WorkMode(mode)
    if isinstance(backend, int):
        backend = Backend(backend)

    if session_id is None:
        session_id = str(uuid.uuid1())
    return create(session_id, mode, backend, options)


def create(session_id=None,
           mode: typing.Union[int, WorkMode] = WorkMode.STANDALONE,
           backend: typing.Union[int, Backend] = Backend.EGGROLL,
           options: dict = None) -> SessionABC:
    if isinstance(mode, int):
        mode = WorkMode(mode)
    if isinstance(backend, int):
        backend = Backend(backend)

    global _DEFAULT_SESSION

    if backend.is_eggroll():
        if mode.is_cluster():
            from fate_arch.session.impl.eggroll import Session
            sess = Session(session_id, work_mode=mode, options=options)
            _DEFAULT_SESSION = sess
            return sess
        else:
            from fate_arch.session.impl.standalone import StandaloneSession
            sess = StandaloneSession(session_id)
            _DEFAULT_SESSION = sess
            return sess
    if backend.is_spark():
        from fate_arch.session.impl.spark import Session
        sess = Session(session_id)
        _DEFAULT_SESSION = sess
        return sess

    raise NotImplementedError()


def has_default():
    return _DEFAULT_SESSION is not None


def default() -> SessionABC:
    if _DEFAULT_SESSION is None:
        raise RuntimeError(f"session not init")
    return _DEFAULT_SESSION


def exit_session():
    global _DEFAULT_SESSION
    _DEFAULT_SESSION = None


def is_table(v):
    return isinstance(v, TableABC)
