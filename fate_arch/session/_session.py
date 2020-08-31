import threading
import typing
import uuid

from fate_arch.abc import CSessionABC, FederationABC, CTableABC
from fate_arch.common import Backend, WorkMode
from fate_arch.computing import ComputingEngine
from fate_arch.federation import FederationEngine
from fate_arch.session._parties import PartiesInfo


class Session(object):

    @staticmethod
    def create(backend: typing.Union[Backend, int] = Backend.EGGROLL,
               work_mode: typing.Union[WorkMode, int] = WorkMode.CLUSTER):
        if isinstance(work_mode, int):
            work_mode = WorkMode(work_mode)
        if isinstance(backend, int):
            backend = Backend(backend)

        if backend == Backend.EGGROLL:
            if work_mode == WorkMode.CLUSTER:
                return Session(ComputingEngine.EGGROLL, FederationEngine.EGGROLL)
            else:
                return Session(ComputingEngine.STANDALONE, FederationEngine.STANDALONE)
        if backend == Backend.SPARK:
            return Session(ComputingEngine.SPARK, FederationEngine.MQ)

    def __init__(self, computing_type: ComputingEngine,
                 federation_type: FederationEngine):
        self._computing_type = computing_type
        self._federation_type = federation_type
        self._computing_session: typing.Optional[CSessionABC] = None
        self._federation_session: typing.Optional[FederationABC] = None
        self._parties_info: typing.Optional[PartiesInfo] = None
        self._session_id = str(uuid.uuid1())

        # add to session environment
        _RuntimeSessionEnvironment.add_session(self)

    @property
    def session_id(self) -> str:
        return self._session_id

    def as_default(self):
        _RuntimeSessionEnvironment.as_default_opened(self)
        return self

    def _open(self):
        _RuntimeSessionEnvironment.open_non_default_session(self)
        return self

    def _close(self):
        _RuntimeSessionEnvironment.close_non_default_session(self)
        return self

    def __enter__(self):
        return self._open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._close()

    def init_computing(self,
                       computing_session_id: str,
                       **kwargs):
        if self.is_computing_valid:
            raise RuntimeError(f"computing session already valid")

        if self._computing_type == ComputingEngine.EGGROLL:
            from fate_arch.computing.eggroll import CSession
            work_mode = kwargs.get("work_mode", 1)
            options = kwargs.get("options", {})
            self._computing_session = CSession(session_id=computing_session_id,
                                               work_mode=work_mode,
                                               options=options)
            return self

        if self._computing_type == ComputingEngine.SPARK:
            from fate_arch.computing.spark import CSession
            self._computing_session = CSession(session_id=computing_session_id)
            self._computing_type = ComputingEngine.SPARK
            return self

        if self._computing_type == ComputingEngine.STANDALONE:
            from fate_arch.computing.standalone import CSession
            self._computing_session = CSession(session_id=computing_session_id)
            self._computing_type = ComputingEngine.STANDALONE
            return self

        raise RuntimeError(f"{self._computing_type} not supported")

    def init_federation(self,
                        federation_session_id: str,
                        *,
                        runtime_conf: typing.Optional[dict] = None,
                        parties_info: typing.Optional[PartiesInfo] = None,
                        service_conf: typing.Optional[dict] = None):

        if parties_info is None:
            if runtime_conf is None:
                raise RuntimeError(f"`party_info` and `runtime_conf` are both `None`")
            parties_info = PartiesInfo.from_conf(runtime_conf)
        self._parties_info = parties_info

        if self.is_federation_valid:
            raise RuntimeError("federation session already valid")

        if self._federation_type == FederationEngine.EGGROLL:
            from fate_arch.computing.eggroll import CSession
            from fate_arch.federation.eggroll import Federation
            from fate_arch.federation.eggroll import Proxy

            if not self.is_computing_valid or not isinstance(self._computing_session, CSession):
                raise RuntimeError(f"require computing with type {ComputingEngine.EGGROLL} valid")

            proxy = Proxy.from_conf(service_conf)
            self._federation_session = Federation(rp_ctx=self._computing_session.get_rpc(),
                                                  rs_session_id=federation_session_id,
                                                  party=parties_info.local_party,
                                                  proxy=proxy)
            return self

        if self._federation_type == FederationEngine.MQ:
            from fate_arch.computing.spark import CSession
            from fate_arch.federation.spark import Federation

            if not self.is_computing_valid or not isinstance(self._computing_session, CSession):
                raise RuntimeError(f"require computing with type {ComputingEngine.SPARK} valid")

            self._federation_session = Federation.from_conf(federation_session_id=federation_session_id,
                                                            party=parties_info.local_party,
                                                            runtime_conf=runtime_conf,
                                                            service_conf=service_conf)
            return self

        if self._federation_type == FederationEngine.STANDALONE:
            from fate_arch.computing.standalone import CSession
            from fate_arch.federation.standalone import Federation

            if not self.is_computing_valid or not isinstance(self._computing_session, CSession):
                raise RuntimeError(f"require computing with type {ComputingEngine.STANDALONE} valid")

            self._federation_session = \
                Federation(standalone_session=self._computing_session.get_standalone_session(),
                           federation_session_id=federation_session_id,
                           party=parties_info.local_party)
            return self

        raise RuntimeError(f"{self._federation_type} not supported")

    @property
    def computing(self) -> CSessionABC:
        return self._computing_session

    @property
    def federation(self) -> FederationABC:
        return self._federation_session

    @property
    def parties(self):
        return self._parties_info

    @property
    def is_computing_valid(self):
        return self._computing_session is not None

    @property
    def is_federation_valid(self):
        return self._federation_session is not None


class _RuntimeSessionEnvironment(object):
    __DEFAULT = None
    __SESSIONS = threading.local()

    @classmethod
    def add_session(cls, session: 'Session'):
        if not hasattr(cls.__SESSIONS, "CREATED"):
            cls.__SESSIONS.CREATED = {}
        cls.__SESSIONS.CREATED[session.session_id] = session

    @classmethod
    def has_non_default_session_opened(cls):
        if getattr(cls.__SESSIONS, 'OPENED_STACK', None) is not None and cls.__SESSIONS.OPENED_STACK:
            return True
        return False

    @classmethod
    def get_non_default_session(cls):
        return cls.__SESSIONS.OPENED_STACK[-1]

    @classmethod
    def open_non_default_session(cls, session):
        if not hasattr(cls.__SESSIONS, 'OPENED_STACK'):
            cls.__SESSIONS.OPENED_STACK = []
        cls.__SESSIONS.OPENED_STACK.append(session)

    @classmethod
    def close_non_default_session(cls, session: Session):
        if not hasattr(cls.__SESSIONS, 'OPENED_STACK') or len(cls.__SESSIONS.OPENED_STACK) == 0:
            raise RuntimeError(f"non_default_session stack empty, nothing to close")
        least: Session = cls.__SESSIONS.OPENED_STACK.pop()
        if least.session_id != session.session_id:
            raise RuntimeError(f"least opened session({least.session_id}) should be close first! "
                               f"while try to close {session.session_id}. all session: {cls.__SESSIONS.OPENED_STACK}")

    @classmethod
    def has_default_session_opened(cls):
        return cls.__DEFAULT is not None

    @classmethod
    def get_default_session(cls):
        return cls.__DEFAULT

    @classmethod
    def as_default_opened(cls, session):
        cls.__DEFAULT = session

    @classmethod
    def get_latest_opened(cls) -> Session:
        if not cls.has_non_default_session_opened():
            if not cls.has_default_session_opened():
                raise RuntimeError(f"no session opened")
            else:
                return cls.get_default_session()
        else:
            return cls.get_non_default_session()


def get_latest_opened() -> Session:
    return _RuntimeSessionEnvironment.get_latest_opened()


# noinspection PyPep8Naming
class computing_session(object):

    @staticmethod
    def init(session_id, work_mode=0, backend=0):
        Session.create(work_mode, backend).init_computing(session_id).as_default()

    @staticmethod
    def parallelize(data: typing.Iterable, partition: int, include_key: bool, **kwargs) -> CTableABC:
        return get_latest_opened().computing.parallelize(data, partition, include_key, **kwargs)

    @staticmethod
    def stop():
        get_latest_opened().computing.stop()


# noinspection PyPep8Naming
class runtime_parties(object):
    @staticmethod
    def roles_to_parties(roles: typing.Iterable, strict=True):
        return get_latest_opened().parties.roles_to_parties(roles=roles, strict=strict)
