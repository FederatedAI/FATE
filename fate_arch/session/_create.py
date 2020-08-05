import typing
import uuid

from fate_arch.common import WorkMode, Backend
from fate_arch.computing import ComputingType
from fate_arch.session._session import Session, set_default


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
           options: dict = None) -> Session:
    if isinstance(mode, int):
        mode = WorkMode(mode)
    if isinstance(backend, int):
        backend = Backend(backend)

    session = Session()
    if backend.is_eggroll():
        computing_type = ComputingType.EGGROLL if mode.is_cluster() else ComputingType.STANDALONE
    elif backend.is_spark():
        computing_type = ComputingType.EGGROLL
    else:
        raise NotImplementedError()
    if options is None:
        options = {}
    session.init_computing(computing_type=computing_type, computing_session_id=session_id, options=options)

    set_default(session)
    return session
