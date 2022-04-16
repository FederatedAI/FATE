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
from fate_arch.abc import CSessionABC, FederationABC, CTableABC, StorageSessionABC, StorageTableABC, StorageTableMetaABC
from fate_arch.common import log, base_utils
from fate_arch.common import remote_status
from fate_arch.computing import ComputingEngine
from fate_arch.federation import FederationEngine
from fate_arch.storage import StorageEngine, StorageSessionBase
from fate_arch.metastore.db_models import DB, SessionRecord, init_database_tables
from fate_arch.session._parties import PartiesInfo

LOGGER = log.getLogger()


class Session(object):
    __GLOBAL_SESSION = None

    @classmethod
    def get_global(cls):
        return cls.__GLOBAL_SESSION

    @classmethod
    def _as_global(cls, sess):
        cls.__GLOBAL_SESSION = sess

    def as_global(self):
        self._as_global(self)
        return self

    def __init__(self, session_id: str = None, options=None):
        if options is None:
            options = {}
        engines = engine_utils.get_engines()
        LOGGER.info(f"using engines: {engines}")
        computing_type = engines.get(EngineType.COMPUTING, None)
        if computing_type is None:
            raise RuntimeError(f"must set default engines on conf/service_conf.yaml")

        self._computing_type = engines.get(EngineType.COMPUTING, None)
        self._federation_type = engines.get(EngineType.FEDERATION, None)
        self._storage_engine = engines.get(EngineType.STORAGE, None)
        self._computing_session: typing.Optional[CSessionABC] = None
        self._federation_session: typing.Optional[FederationABC] = None
        self._storage_session: typing.Dict[StorageSessionABC] = {}
        self._parties_info: typing.Optional[PartiesInfo] = None
        self._session_id = str(uuid.uuid1()) if not session_id else session_id
        self._logger = LOGGER if options.get("logger", None) is None else options.get("logger", None)

        self._logger.info(f"create manager session {self._session_id}")

        # init meta db
        init_database_tables()

    @property
    def session_id(self) -> str:
        return self._session_id

    def _open(self):
        return self

    def _close(self):
        self.destroy_all_sessions()

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

        if self._computing_type == ComputingEngine.STANDALONE:
            from fate_arch.computing.standalone import CSession

            options = kwargs.get("options", {})
            self._computing_session = CSession(session_id=computing_session_id, options=options)
            self._computing_type = ComputingEngine.STANDALONE
            return self

        if self._computing_type == ComputingEngine.EGGROLL:
            from fate_arch.computing.eggroll import CSession

            options = kwargs.get("options", {})
            self._computing_session = CSession(
                session_id=computing_session_id, options=options
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

        raise RuntimeError(f"{self._federation_type} not supported")

    def _get_or_create_storage(self,
                               storage_session_id=None,
                               storage_engine=None,
                               record: bool = True,
                               **kwargs) -> StorageSessionABC:
        storage_session_id = f"{self._session_id}_storage_{uuid.uuid1()}" if not storage_session_id else storage_session_id

        if storage_session_id in self._storage_session:
            return self._storage_session[storage_session_id]
        else:
            if storage_engine is None:
                storage_engine = self._storage_engine

        for session in self._storage_session.values():
            if storage_engine == session.engine:
                return session

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

        elif storage_engine == StorageEngine.PATH:
            from fate_arch.storage.path import StorageSession
            storage_session = StorageSession(session_id=storage_session_id, options=kwargs.get("options", {}))

        elif storage_engine == StorageEngine.LOCALFS:
            from fate_arch.storage.localfs import StorageSession
            storage_session = StorageSession(session_id=storage_session_id, options=kwargs.get("options", {}))

        else:
            raise NotImplementedError(f"can not be initialized with storage engine: {storage_engine}")

        self._storage_session[storage_session_id] = storage_session

        return storage_session

    def get_table(self, name, namespace, ignore_disable=False) -> typing.Union[StorageTableABC, None]:
        meta = Session.get_table_meta(name=name, namespace=namespace)
        if meta is None:
            return None
        if meta.get_disable() and not ignore_disable:
            raise Exception(f"table {namespace} {name} disable: {meta.get_disable()}")
        engine = meta.get_engine()
        storage_session = self._get_or_create_storage(storage_engine=engine)
        table = storage_session.get_table(name=name, namespace=namespace)
        return table

    @classmethod
    def get_table_meta(cls, name, namespace) -> typing.Union[StorageTableMetaABC, None]:
        meta = StorageSessionBase.get_table_meta(name=name, namespace=namespace)
        return meta

    @classmethod
    def persistent(cls, computing_table: CTableABC, namespace, name, schema=None, part_of_data=None,
                   engine=None, engine_address=None, store_type=None, token: typing.Dict = None) -> StorageTableMetaABC:
        return StorageSessionBase.persistent(computing_table=computing_table,
                                             namespace=namespace,
                                             name=name,
                                             schema=schema,
                                             part_of_data=part_of_data,
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
        return self._get_or_create_storage(**kwargs)

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
        self._logger.info(
            f"try to save session record for manager {self._session_id}, {engine_type} {engine_name} {engine_session_id}")
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
        self._logger.info(
            f"save session record for manager {self._session_id}, {engine_type} {engine_name} {engine_session_id} successfully")

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
    def get_session_from_record(self, **kwargs):
        self._logger.info(f"query by manager session id {self._session_id}")
        session_records = self.query_sessions(manager_session_id=self._session_id, **kwargs)
        self._logger.info([session_record.f_engine_session_id for session_record in session_records])
        for session_record in session_records:
            try:
                engine_session_id = session_record.f_engine_session_id
                if session_record.f_engine_type == EngineType.COMPUTING:
                    self._init_computing_if_not_valid(computing_session_id=engine_session_id)
                elif session_record.f_engine_type == EngineType.STORAGE:
                    self._get_or_create_storage(storage_session_id=engine_session_id,
                                                storage_engine=session_record.f_engine_name,
                                                record=False)
            except Exception as e:
                self._logger.error(e)
                self.delete_session_record(engine_session_id=session_record.f_engine_session_id)

    def _init_computing_if_not_valid(self, computing_session_id):
        if not self.is_computing_valid:
            self.init_computing(computing_session_id=computing_session_id, record=False)
            return True
        elif self._computing_session.session_id != computing_session_id:
            self._logger.warning(
                f"manager session had computing session {self._computing_session.session_id} different with query from db session {computing_session_id}")
            return False
        else:
            # already exists
            return True

    def destroy_all_sessions(self, **kwargs):
        self._logger.info(f"start destroy manager session {self._session_id} all sessions")
        self.get_session_from_record(**kwargs)
        self.destroy_storage_session()
        self.destroy_computing_session()
        self._logger.info(f"finish destroy manager session {self._session_id} all sessions")

    def destroy_computing_session(self):
        if self.is_computing_valid:
            try:
                self._logger.info(f"try to destroy computing session {self._computing_session.session_id}")
                try:
                    ret = self._computing_session.stop()
                except BaseException:
                    ret = self._computing_session.kill()
                self._logger.info(f"destroy computing session {self._computing_session.session_id} successfully, "
                                  f"ret={ret}")
            except Exception as e:
                self._logger.info(f"destroy computing session {self._computing_session.session_id} failed", e)
            self.delete_session_record(engine_session_id=self._computing_session.session_id)

    def destroy_storage_session(self):
        for session_id, session in self._storage_session.items():
            try:
                self._logger.info(f"try to destroy storage session {session_id}")
                session.destroy()
                self._logger.info(f"destroy storage session {session_id} successfully")
            except Exception as e:
                self._logger.exception(f"destroy storage session {session_id} failed", e)
            self.delete_session_record(engine_session_id=session_id)

    def wait_remote_all_done(self, timeout=None):
        LOGGER.info(f"remote futures: {remote_status._remote_futures}, waiting...")
        remote_status.wait_all_remote_done(timeout)
        LOGGER.info(f"remote futures: {remote_status._remote_futures}, all done")


def get_session() -> Session:
    return Session.get_global()


def get_parties() -> PartiesInfo:
    return get_session().parties


def get_computing_session() -> CSessionABC:
    return get_session().computing

# noinspection PyPep8Naming


class computing_session(object):
    @staticmethod
    def init(session_id, options=None):
        Session(options=options).as_global().init_computing(session_id)

    @staticmethod
    def parallelize(data: typing.Iterable, partition: int, include_key: bool, **kwargs) -> CTableABC:
        return get_computing_session().parallelize(data, partition=partition, include_key=include_key, **kwargs)

    @staticmethod
    def stop():
        return get_computing_session().stop()
