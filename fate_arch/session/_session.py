import typing

from fate_arch.abc import CSessionABC, FederationABC
from fate_arch.common.file_utils import load_json_conf
from fate_arch.computing import ComputingType
from fate_arch.storage import StorageType
from fate_arch.federation import FederationType
from fate_arch.session._parties import Parties


class Session(object):
    def __init__(self):
        self._computing_session: typing.Optional[CSessionABC] = None
        self._computing_type: typing.Optional[ComputingType] = None
        self._federation_session: typing.Optional[FederationABC] = None
        self._parties: typing.Optional[Parties] = None
        self._storage_session = None

    def init_computing(self,
                       computing_type: ComputingType,
                       computing_session_id: str,
                       **kwargs):
        if self.is_computing_valid:
            raise RuntimeError(f"computing session already valid")

        if computing_type == ComputingType.EGGROLL:
            from fate_arch.computing.eggroll import CSession
            work_mode = kwargs.get("work_mode", 1)
            options = kwargs.get("options", {})
            self._computing_session = CSession(session_id=computing_session_id,
                                               work_mode=work_mode,
                                               options=options)
            self._computing_type = ComputingType.EGGROLL
            return self

        if computing_type == ComputingType.SPARK:
            from fate_arch.computing.spark import CSession
            self._computing_session = CSession(session_id=computing_session_id)
            self._computing_type = ComputingType.SPARK
            return self

        if computing_type == ComputingType.STANDALONE:
            from fate_arch.computing.standalone import CSession
            self._computing_session = CSession(session_id=computing_session_id)
            self._computing_type = ComputingType.STANDALONE
            return self

        raise RuntimeError(f"{computing_type} not supported")

    def init_federation(self,
                        federation_session_id: str,
                        runtime_conf: dict,
                        server_conf: dict = None,
                        federation_type: typing.Optional[FederationType] = None):
        self._parties = Parties.from_runtime_conf(runtime_conf)
        party = self._parties.local_party
        if server_conf is None:
            server_conf = load_json_conf("conf/server_conf.json")

        if self.is_federation_valid:
            raise RuntimeError("federation session already valid")

        if federation_type is None:
            if self._computing_type is None:
                raise RuntimeError("can't infer federation_type since session not valid")

            if self._computing_type == ComputingType.EGGROLL:
                federation_type = FederationType.EGGROLL
            elif self._computing_type == ComputingType.SPARK:
                federation_type = FederationType.MQ
            elif self._computing_type == ComputingType.STANDALONE:
                federation_type = FederationType.STANDALONE
            else:
                raise RuntimeError(f"can't infer federation_type with computing type {self._computing_type}")

        if federation_type == FederationType.EGGROLL:
            from fate_arch.computing.eggroll import CSession
            from fate_arch.federation.eggroll import Federation
            from fate_arch.federation.eggroll import Proxy

            if not self.is_computing_valid or not isinstance(self._computing_session, CSession):
                raise RuntimeError(f"require computing with type {ComputingType.EGGROLL} valid")

            proxy = Proxy.from_conf(server_conf)
            self._federation_session = Federation(rp_ctx=self._computing_session.get_rpc(),
                                                  rs_session_id=federation_session_id,
                                                  party=party,
                                                  proxy=proxy)
            return self

        if federation_type == FederationType.MQ:
            from fate_arch.computing.spark import CSession
            from fate_arch.federation.spark import Federation

            if not self.is_computing_valid or not isinstance(self._computing_session, CSession):
                raise RuntimeError(f"require computing with type {ComputingType.SPARK} valid")

            self._federation_session = Federation.from_conf(federation_session_id=federation_session_id,
                                                            party=party,
                                                            runtime_conf=runtime_conf,
                                                            server_conf=server_conf)
            return self

        if federation_type == FederationType.STANDALONE:
            from fate_arch.computing.standalone import CSession
            from fate_arch.federation.standalone import Federation

            if not self.is_computing_valid or not isinstance(self._computing_session, CSession):
                raise RuntimeError(f"require computing with type {ComputingType.STANDALONE} valid")

            self._federation_session = \
                Federation(standalone_session=self._computing_session.get_standalone_session(),
                           federation_session_id=federation_session_id,
                           party=party)
            return self

        raise RuntimeError(f"{federation_type} not supported")

    def init_storage(self, storage_type: StorageType = FederationType.EGGROLL):
        pass

    @property
    def computing(self) -> CSessionABC:
        return self._computing_session

    @property
    def federation(self) -> FederationABC:
        return self._federation_session

    @property
    def storage(self):
        return self._storage_session

    @property
    def parties(self):
        return self._parties

    @property
    def is_computing_valid(self):
        return self._computing_session is not None

    @property
    def is_federation_valid(self):
        return self._federation_session is not None

    @property
    def is_storage_valid(self):
        return self._storage_session is not None
