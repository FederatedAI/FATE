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
import threading
import typing
import uuid

import peewee
from fate_arch.common import engine_utils, EngineType
from fate_arch.relation_ship import Relationship
from fate_arch.abc import CSessionABC, FederationABC, CTableABC, StorageSessionABC
from fate_arch.common import log, base_utils
from fate_arch.common import Backend, WorkMode, remote_status
from fate_arch.computing import ComputingEngine
from fate_arch.federation import FederationEngine
from fate_arch.storage import StorageEngine, StorageSessionBase, StorageTableMeta
from fate_arch.metastore.db_models import DB, SessionRecord, init_database_tables
from fate_arch.session._parties import PartiesInfo

LOGGER = log.getLogger()


class Session(object):
    @staticmethod
    def create(backend: typing.Union[Backend, int] = None,
               work_mode: typing.Union[WorkMode, int] = None, **kwargs):
        new_kwargs = locals().copy()
        new_kwargs.update(kwargs)
        engines = engine_utils.engines_compatibility(**new_kwargs)
        LOGGER.info(f"using engines: {engines}")
        return Session(**engines)

    def __init__(self, computing: ComputingEngine = None, federation: FederationEngine = None, storage: StorageEngine = None, session_id: str = None, logger=None, **kwargs):
        self._computing_type = computing
        self._federation_type = federation
        self._storage_engine = storage
        self._computing_session: typing.Optional[CSessionABC] = None
        self._federation_session: typing.Optional[FederationABC] = None
        self._storage_session: typing.Dict[StorageSessionABC] = {}
        self._parties_info: typing.Optional[PartiesInfo] = None
        self._session_id = str(uuid.uuid1()) if not session_id else session_id
        self._logger = LOGGER if logger is None else logger

        self._logger.info(f"create manager session {self._session_id}")

        # add to session environment
        _RuntimeSessionEnvironment.add_session(self)

        self._logger.info(f"thread environment had sessions {_RuntimeSessionEnvironment.get_all()}")

        # init meta db
        init_database_tables()

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
        if exc_tb:
            self._logger.exception("", exc_info=(exc_type, exc_val, exc_tb))
        return self._close()

    def init_computing(self,
                       computing_session_id: str = None,
                       record: bool = True,
                       **kwargs):
        computing_session_id = f"{self._session_id}_computing_{uuid.uuid1()}" if not computing_session_id else computing_session_id
        if self.is_computing_valid:
            raise RuntimeError(f"computing session already valid")

        if record:
            self.save_record(engine_type=EngineType.COMPUTING,
                             engine_name=self._computing_type,
                             engine_session_id=computing_session_id)

        if self._computing_type == ComputingEngine.EGGROLL:
            from fate_arch.computing.eggroll import CSession

            work_mode = kwargs.get("work_mode", WorkMode.CLUSTER)
            options = kwargs.get("options", {})
            self._computing_session = CSession(
                session_id=computing_session_id, work_mode=work_mode, options=options
            )
            return self

        if self._computing_type == ComputingEngine.SPARK:
            from fate_arch.computing.spark import CSession

            self._computing_session = CSession(session_id=computing_session_id)
            self._computing_type = ComputingEngine.SPARK
            return self

        if self._computing_type == ComputingEngine.LINKIS_SPARK:
            from fate_arch.computing.spark import CSession
            self._computing_session = CSession(session_id=computing_session_id)
            self._computing_type = ComputingEngine.LINKIS_SPARK
            return self

        if self._computing_type == ComputingEngine.STANDALONE:
            from fate_arch.computing.standalone import CSession

            self._computing_session = CSession(session_id=computing_session_id)
            self._computing_type = ComputingEngine.STANDALONE
            return self

        raise RuntimeError(f"{self._computing_type} not supported")

    def init_federation(
        self,
        federation_session_id: str,
        *,
        runtime_conf: typing.Optional[dict] = None,
        parties_info: typing.Optional[PartiesInfo] = None,
        service_conf: typing.Optional[dict] = None,
    ):

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

            if not self.is_computing_valid or not isinstance(
                self._computing_session, CSession
            ):
                raise RuntimeError(
                    f"require computing with type {ComputingEngine.EGGROLL} valid"
                )

            self._federation_session = Federation(
                rp_ctx=self._computing_session.get_rpc(),
                rs_session_id=federation_session_id,
                party=parties_info.local_party,
                proxy_endpoint=f"{service_conf['host']}:{service_conf['port']}",
            )
            return self

        if self._federation_type == FederationEngine.RABBITMQ:
            from fate_arch.computing.spark import CSession
            from fate_arch.federation.rabbitmq import Federation

            if not self.is_computing_valid or not isinstance(
                self._computing_session, CSession
            ):
                raise RuntimeError(
                    f"require computing with type {ComputingEngine.SPARK} valid"
                )

            self._federation_session = Federation.from_conf(
                federation_session_id=federation_session_id,
                party=parties_info.local_party,
                runtime_conf=runtime_conf,
                rabbitmq_config=service_conf,
            )
            return self

        # Add pulsar support
        if self._federation_type == FederationEngine.PULSAR:
            from fate_arch.computing.spark import CSession
            from fate_arch.federation.pulsar import Federation

            if not self.is_computing_valid or not isinstance(
                self._computing_session, CSession
            ):
                raise RuntimeError(
                    f"require computing with type {ComputingEngine.SPARK} valid"
                )

            self._federation_session = Federation.from_conf(
                federation_session_id=federation_session_id,
                party=parties_info.local_party,
                runtime_conf=runtime_conf,
                pulsar_config=service_conf,
            )
            return self

        if self._federation_type == FederationEngine.STANDALONE:
            from fate_arch.computing.standalone import CSession
            from fate_arch.federation.standalone import Federation

            if not self.is_computing_valid or not isinstance(
                self._computing_session, CSession
            ):
                raise RuntimeError(
                    f"require computing with type {ComputingEngine.STANDALONE} valid"
                )

            self._federation_session = Federation(
                standalone_session=self._computing_session.get_standalone_session(),
                federation_session_id=federation_session_id,
                party=parties_info.local_party,
            )
            return self

        raise RuntimeError(f"{self._federation_type} not supported")

    def new_storage(self, storage_session_id=None, storage_engine=None, computing_engine=None, record: bool = True, **kwargs):
        storage_session_id = f"{self._session_id}_storage_{uuid.uuid1()}" if not storage_session_id else storage_session_id
        if storage_session_id in self._storage_session:
            raise RuntimeError(f"the storage session id {storage_session_id} already exists")
        if kwargs.get("name") and kwargs.get("namespace"):
            storage_engine, address, partitions = StorageSessionBase.get_storage_info(name=kwargs.get("name"),
                                                                                      namespace=kwargs.get("namespace"))
            if not storage_engine:
                return None
        if storage_engine is None and computing_engine is None:
            computing_engine, federation_engine, federation_mode = engine_utils.engines_compatibility(**kwargs)
        if storage_engine is None and computing_engine:
            # Gets the computing engine default storage engine
            storage_engine = Relationship.Computing.get(computing_engine, {}).get(EngineType.STORAGE, {}).get("default", None)

        if record:
            self.save_record(engine_type=EngineType.STORAGE,
                             engine_name=storage_engine,
                             engine_session_id=storage_session_id)

        if storage_engine == StorageEngine.EGGROLL:
            from fate_arch.storage.eggroll import StorageSession
            storage_session = StorageSession(session_id=storage_session_id, options=kwargs.get("options", {}))
        elif storage_engine == StorageEngine.STANDALONE:
            from fate_arch.storage.standalone import StorageSession
            storage_session = StorageSession(session_id=storage_session_id, options=kwargs.get("options", {}))
        elif storage_engine == StorageEngine.MYSQL:
            from fate_arch.storage.mysql import StorageSession
            storage_session = StorageSession(session_id=storage_session_id, options=kwargs.get("options", {}))
        elif storage_engine == StorageEngine.HDFS:
            from fate_arch.storage.hdfs import StorageSession
            storage_session = StorageSession(session_id=storage_session_id, options=kwargs.get("options", {}))
        elif storage_engine == StorageEngine.HIVE:
            from fate_arch.storage.hive import StorageSession
            storage_session = StorageSession(session_id=storage_session_id, options=kwargs.get("options", {}))
        elif storage_engine == StorageEngine.LINKIS_HIVE:
            from fate_arch.storage.linkis_hive import StorageSession
            storage_session = StorageSession(session_id=storage_session_id, options=kwargs.get("options", {}))
        elif storage_engine == StorageEngine.FILE:
            from fate_arch.storage.file import StorageSession
            storage_session = StorageSession(session_id=storage_session_id, options=kwargs.get("options", {}))
        elif storage_engine == StorageEngine.PATH:
            from fate_arch.storage.path import StorageSession
            storage_session = StorageSession(session_id=storage_session_id, options=kwargs.get("options", {}))
        else:
            raise NotImplementedError(f"can not be initialized with storage engine: {storage_engine}")
        if kwargs.get("name") and kwargs.get("namespace"):
            storage_session.set_default(name=kwargs["name"], namespace=kwargs["namespace"])
        self._storage_session[storage_session_id] = storage_session
        return storage_session

    @classmethod
    def persistent(cls, computing_table: CTableABC, table_namespace, table_name, schema=None, engine=None, engine_address=None, store_type=None, token: typing.Dict = None) -> StorageTableMeta:
        return StorageSessionBase.persistent(computing_table=computing_table,
                                             table_namespace=table_namespace,
                                             table_name=table_name,
                                             schema=schema,
                                             engine=engine,
                                             engine_address=engine_address,
                                             store_type=store_type,
                                             token=token)

    @property
    def computing(self) -> CSessionABC:
        return self._computing_session

    @property
    def federation(self) -> FederationABC:
        return self._federation_session

    def storage(self, **kwargs):
        return self.new_storage(**kwargs)

    @property
    def parties(self):
        return self._parties_info

    @property
    def is_computing_valid(self):
        return self._computing_session is not None

    @property
    def is_federation_valid(self):
        return self._federation_session is not None

    @DB.connection_context()
    def save_record(self, engine_type, engine_name, engine_session_id):
        self._logger.info(f"try to save session record for manager {self._session_id}, {engine_type} {engine_name} {engine_session_id}")
        session_record = SessionRecord()
        session_record.f_manager_session_id = self._session_id
        session_record.f_engine_type = engine_type
        session_record.f_engine_name = engine_name
        session_record.f_engine_session_id = engine_session_id
        # TODO: engine address
        session_record.f_engine_address = {}
        session_record.f_create_time = base_utils.current_timestamp()
        msg = f"save storage session record for manager {self._session_id}, {engine_type} {engine_name} {engine_session_id}"
        try:
            effect_count = session_record.save(force_insert=True)
            if effect_count != 1:
                raise RuntimeError(f"{msg} failed")
        except peewee.IntegrityError as e:
            LOGGER.warning(e)
        except Exception as e:
            raise RuntimeError(f"{msg} exception", e)
        self._logger.info(f"save session record for manager {self._session_id}, {engine_type} {engine_name} {engine_session_id} successfully")

    @DB.connection_context()
    def delete_session_record(self, engine_session_id):
        rows = SessionRecord.delete().where(SessionRecord.f_engine_session_id == engine_session_id).execute()
        if rows > 0:
            self._logger.info(f"delete session {engine_session_id} record successfully")
        else:
            self._logger.warning(f"delete session {engine_session_id} record failed")

    @classmethod
    @DB.connection_context()
    def query_sessions(cls, reverse=None, order_by=None, **kwargs):
        return SessionRecord.query(reverse=reverse, order_by=order_by, **kwargs)

    @DB.connection_context()
    def get_session_from_record(self):
        self._logger.info(f"query by manager session id {self._session_id}")
        session_records = self.query_sessions(manager_session_id=self._session_id)
        self._logger.info([session_record.f_engine_session_id for session_record in session_records])
        for session_record in session_records:
            engine_session_id = session_record.f_engine_session_id
            if session_record.f_engine_type == EngineType.COMPUTING:
                self.add_computing(computing_session_id=engine_session_id)
            elif session_record.f_engine_type == EngineType.STORAGE:
                self.add_storage(storage_session_id=engine_session_id, storage_engine=session_record.f_engine_name)

    def add_computing(self, computing_session_id):
        if not self.is_computing_valid:
            self.init_computing(computing_session_id=computing_session_id, record=False)
            return True
        elif self._computing_session.session_id != computing_session_id:
            self._logger.warning(f"manager session had computing session {self._computing_session.session_id} different with query from db session {computing_session_id}")
            return False
        else:
            # already exists
            return True

    def add_storage(self, storage_session_id, storage_engine):
        if storage_session_id not in self._storage_session:
            self.new_storage(storage_session_id=storage_session_id,
                             storage_engine=storage_engine,
                             record=False)
        return True

    def destroy_all_sessions(self):
        self._logger.info(f"start destroy manager session {self._session_id} all sessions")
        self.get_session_from_record()
        self.destroy_storage()
        self.destroy_computing()
        self._logger.info(f"finish destroy manager session {self._session_id} all sessions")

    def destroy_computing(self):
        if self.is_computing_valid:
            try:
                self._logger.info(f"try to destroy computing session {self._computing_session.session_id}")
                try:
                    self._computing_session.stop()
                except:
                    self._computing_session.kill()
                self._logger.info(f"destroy computing session {self._computing_session.session_id} successfully")
                self.delete_session_record(engine_session_id=self._computing_session.session_id)
            except Exception as e:
                self._logger.info(f"destroy computing session {self._computing_session.session_id} failed", e)

    def destroy_storage(self):
        for session_id, session in self._storage_session.items():
            try:
                self._logger.info(f"try to destroy storage session {session_id}")
                session.destroy()
                self._logger.info(f"destroy storage session {session_id} successfully")
                self.delete_session_record(engine_session_id=session_id)
            except Exception as e:
                self._logger.exception(f"destroy storage session {session_id} failed", e)

    def wait_remote_all_done(self, timeout=None):
        LOGGER.info(f"remote futures: {remote_status._remote_futures}, waiting...")
        remote_status.wait_all_remote_done(timeout)
        LOGGER.info(f"remote futures: {remote_status._remote_futures}, all done")


class _RuntimeSessionEnvironment(object):
    __DEFAULT = None
    __SESSIONS = threading.local()

    @classmethod
    def get_all(cls):
        return cls.__SESSIONS.CREATED

    @classmethod
    def add_session(cls, session: "Session"):
        if not hasattr(cls.__SESSIONS, "CREATED"):
            cls.__SESSIONS.CREATED = {}
        cls.__SESSIONS.CREATED[session.session_id] = session

    @classmethod
    def has_non_default_session_opened(cls):
        if (
            getattr(cls.__SESSIONS, "OPENED_STACK", None) is not None
            and cls.__SESSIONS.OPENED_STACK
        ):
            return True
        return False

    @classmethod
    def get_non_default_session(cls):
        return cls.__SESSIONS.OPENED_STACK[-1]

    @classmethod
    def open_non_default_session(cls, session):
        if not hasattr(cls.__SESSIONS, "OPENED_STACK"):
            cls.__SESSIONS.OPENED_STACK = []
        cls.__SESSIONS.OPENED_STACK.append(session)

    @classmethod
    def close_non_default_session(cls, session: Session):
        if (
            not hasattr(cls.__SESSIONS, "OPENED_STACK")
            or len(cls.__SESSIONS.OPENED_STACK) == 0
        ):
            raise RuntimeError(f"non_default_session stack empty, nothing to close")
        least: Session = cls.__SESSIONS.OPENED_STACK.pop()
        cls.__SESSIONS.CREATED.pop(session.session_id)
        if least.session_id != session.session_id:
            raise RuntimeError(
                f"least opened session({least.session_id}) should be close first! "
                f"while try to close {session.session_id}. all session: {cls.__SESSIONS.OPENED_STACK}"
            )

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
        """
        initialize a computing session

        Parameters
        ----------
        session_id: str
           session id
        work_mode: int
           work mode, 0 for standalone, 1 for cluster
        backend: int
           computing backend, 0 for eggroll, 1 for spark

        Returns
        -------
        instance of concrete subclass of ``CSessionABC``
           computing session
        """
        Session.create(work_mode=work_mode, backend=backend).init_computing(
            session_id
        ).as_default()

    @staticmethod
    def parallelize(
        data: typing.Iterable, partition: int, include_key: bool, **kwargs
    ) -> CTableABC:
        """
        create table from iterable data

        Parameters
        ----------
        data: Iterable
           data to create table from
        partition: int
           number of partitions of created table
        include_key: bool
           ``True`` for create table directly from data, ``False`` for create table with generated keys start from 0

        Returns
        -------
        instance of concrete subclass fo ``CTableABC``
           a table create from data

        """
        return get_latest_opened().computing.parallelize(
            data, partition=partition, include_key=include_key, **kwargs
        )

    @staticmethod
    def stop():
        """
        stop session
        """
        get_latest_opened().computing.stop()


# noinspection PyPep8Naming
class runtime_parties(object):
    @staticmethod
    def roles_to_parties(roles: typing.Iterable, strict=True):
        return get_latest_opened().parties.roles_to_parties(roles=roles, strict=strict)
