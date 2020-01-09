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
import abc
from collections import deque
from typing import Tuple, Union

from arch.api.utils import file_utils
from arch.api.utils.log_utils import getLogger

__all__ = ["Rubbish", "Cleaner", "Party", "init", "FederationWrapped", "Federation", "FederationAuthorization", "ROLES"]

ROLES = ["arbiter", "guest", "host"]
TRANSFER_CONF_PATH = "federatedml/transfer_variable/definition/transfer_conf.json"
CONF_KEY_LOCAL = "local"
CONF_KEY_FEDERATION = "federation"
CONF_KEY_SERVER = "servers"
LOGGER = getLogger()


class Party(object):
    """
    Uniquely identify
    """

    def __init__(self, role, party_id):
        self.role = role
        self.party_id = party_id

    def __hash__(self):
        return (self.role, self.party_id).__hash__()

    def __str__(self):
        return f"Party(role={self.role}, party_id={self.party_id})"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return (self.role, self.party_id) < (other.role, other.party_id)

    def __eq__(self, other):
        return self.party_id == other.party_id and self.role == other.role

    def to_pb(self):
        from arch.api.proto import federation_pb2
        return federation_pb2.Party(partyId=f"{self.party_id}", name=self.role)


class Rubbish(object):
    """
    a collection collects all tables / objects in federation tagged by `tag`.
    """

    def __init__(self, name, tag):
        self._name = name
        self._tag = tag
        self._tables = []
        self._kv = {}

    @property
    def tag(self):
        return self._tag

    def add_table(self, table):
        self._tables.append(table)

    # noinspection PyProtectedMember
    def add_obj(self, table, key):
        if (table._name, table._namespace) not in self._kv:
            self._kv[(table._name, table._namespace)] = (table, [key])
        else:
            self._kv[(table._name, table._namespace)][1].append(key)

    def merge(self, rubbish: 'Rubbish'):
        self._tables.extend(rubbish._tables)
        for tk, (t, k) in rubbish._kv.items():
            if tk in self._kv:
                self._kv[tk][1].extend(k)
            else:
                self._kv[tk] = (t, k)
        # # warm: this is necessary to prevent premature clean work invoked by `__del__` in `rubbish`
        # rubbish.empty()
        return self

    def empty(self):
        self._tables = []
        self._kv = {}

    # noinspection PyBroadException
    def clean(self):
        if self._tables or self._kv:
            LOGGER.debug(f"[CLEAN] {self._name} cleaning rubbishes tagged by {self._tag}")
        for table in self._tables:
            try:
                LOGGER.debug(f"[CLEAN] try destroy table {table}")
                table.destroy()
            except:
                pass

        for _, (table, keys) in self._kv.items():
            for key in keys:
                try:
                    LOGGER.debug(f"[CLEAN] try delete object with key={key} from table={table}")
                    table.delete(key)
                except:
                    pass

    # def __del__(self):  # this is error prone, please call `clean` explicit
    #     self.clean()


class Cleaner(object):
    def __init__(self):
        self._ashcan: deque[Rubbish] = deque()

    def push(self, rubbish):
        """
        append `rubbish`
        :param rubbish: a rubbish instance
        :return:
        """
        if self.is_latest_tag(rubbish.tag):
            self._ashcan[-1].merge(rubbish)
        else:
            self._ashcan.append(rubbish)
        return self

    def is_latest_tag(self, tag):
        return len(self._ashcan) > 0 and self._ashcan[-1].tag == tag

    def keep_latest_n(self, n):
        while len(self._ashcan) > n:
            self._ashcan.popleft().clean()  # clean explicit

    def clean_all(self):
        while len(self._ashcan) > 0:
            self._ashcan.popleft().clean()


class FederationWrapped(object):
    """
    A wrapper, wraps _DTable as Table
    """

    # noinspection PyProtectedMember
    def __init__(self, session_id, work_mode, table_cls):

        if work_mode.is_standalone():
            from eggroll.api.standalone.eggroll import _DTable
            self.dtable_cls = _DTable
        elif work_mode.is_cluster():
            from eggroll.api.cluster.eggroll import _DTable
            self.dtable_cls = _DTable
        else:
            raise EnvironmentError(f"{work_mode} unknown")

        self.table_cls = table_cls
        self._session_id = session_id

    def unboxed(self, obj):
        if isinstance(obj, self.table_cls):
            return obj.dtable()
        else:
            return obj

    def boxed(self, obj):
        if isinstance(obj, self.dtable_cls):
            return self.table_cls.from_dtable(dtable=obj, session_id=self._session_id)
        else:
            return obj


class FederationAuthorization(object):

    def __init__(self, transfer_conf_path):
        self.transfer_auth = file_utils.load_json_conf(transfer_conf_path)

        # cache
        self._authorized_src = {}
        self._authorized_dst = {}

    def _update_auth(self, variable_name):
        a_name, v_name = variable_name.split(".")
        variable_auth = self.transfer_auth.get(a_name, {}).get(v_name, None)
        if variable_auth is None:
            raise ValueError(f"Unauthorized variable: {v_name}")
        auth_src = variable_auth["src"]
        if not isinstance(auth_src, list):
            auth_src = [auth_src]
        auth_dst = variable_auth["dst"]
        self._authorized_src[variable_name] = auth_src
        self._authorized_dst[variable_name] = auth_dst

    def authorized_src_roles(self, variable_name):
        if variable_name not in self._authorized_src:
            self._update_auth(variable_name)
        return self._authorized_src[variable_name]

    def authorized_dst_roles(self, variable_name):
        if variable_name not in self._authorized_dst:
            self._update_auth(variable_name)
        return self._authorized_dst[variable_name]


class Federation(object):

    def __init__(self, session_id, runtime_conf):
        if CONF_KEY_LOCAL not in runtime_conf:
            raise EnvironmentError("runtime_conf should be a dict containing key: {}".format(CONF_KEY_LOCAL))

        self._role = runtime_conf.get(CONF_KEY_LOCAL).get("role")
        self._party_id = runtime_conf.get(CONF_KEY_LOCAL).get("party_id")
        self._local_party = Party(self._role, self._party_id)
        self._session_id = session_id
        self._authorize = FederationAuthorization(TRANSFER_CONF_PATH)
        self._role_to_parties_map = {}
        self._all_parties = []
        for role in ROLES:
            party_id_list = runtime_conf.get('role').get(role, [])
            role_parties = [Party(role, party_id) for party_id in party_id_list]
            self._role_to_parties_map[role] = role_parties
            self._all_parties.extend(role_parties)

    @property
    def local_party(self):
        return self._local_party

    @property
    def all_parties(self):
        return self._all_parties

    def roles_to_parties(self, roles: list) -> list:
        nodes = []
        for role in roles:
            nodes.extend(self._role_to_parties_map[role])
        return nodes

    def role_to_party(self, role, idx) -> Party:
        return self._role_to_parties_map[role][idx]

    def authorized_src_roles(self, name) -> list:
        return self._authorize.authorized_src_roles(name)

    def authorized_dst_roles(self, name) -> list:
        return self._authorize.authorized_dst_roles(name)

    def _get_side_auth(self, name, parties):
        # check auth
        if self._role not in self._authorize.authorized_dst_roles(variable_name=name):
            raise PermissionError(f"try to get obj to {self._role}, with variable named {name}")
        roles = {party.role for party in parties}
        for role in roles:
            if role not in self._authorize.authorized_src_roles(variable_name=name):
                raise PermissionError(f"try to get obj from {role}, with variable named {name}")

    def _remote_side_auth(self, name, parties):
        # check auth
        if self._role not in self._authorize.authorized_src_roles(variable_name=name):
            raise PermissionError(f"try to remote obj from {self._role}, with variable named {name}")
        roles = {party.role for party in parties}
        for role in roles:
            if role not in self._authorize.authorized_dst_roles(variable_name=name):
                raise PermissionError(f"try to remote obj to {role}, with variable named {name}")

    @abc.abstractmethod
    def remote(self, obj, name: str, tag: str, parties: Union[Party, list]) -> Rubbish:
        """
        remote object or dtable to parties identified by `parties`,
        through transfer_variable identified by name and tag.

        :param obj: dtable or object to be remote
        :param name: name of transfer_variable
        :param tag: tag of transfer_variable
        :param parties: instance or instances of `Party`, specify the parties to send obj to
        :return: an instance of Cleaner for latter clean tasks.
        """
        pass

    @abc.abstractmethod
    def get(self, name: str, tag: str, parties: Union[Party, list]) -> Tuple[list, Rubbish]:
        """
         get object or dtable from parties identified by `parties`,
         through transfer_variable identified by name and tag.
        :param name: name of transfer_variable
        :param tag: tag of transfer_variable
        :param parties: instance of Party or list of instances of parties, specify the parties to get obj from
        :return: a tuple, with a list of results and an instance of Cleaner for latter clean tasks.
        """
        pass


def init_table_wrapper(session_id, work_mode, backend):
    if backend.is_eggroll():
        from arch.api.table.eggroll.table_impl import DTable
        return FederationWrapped(session_id=session_id,
                                 work_mode=work_mode,
                                 table_cls=DTable)
    elif backend.is_spark():
        from arch.api.table.pyspark.table_impl import RDDTable
        return FederationWrapped(session_id=session_id,
                                 work_mode=work_mode,
                                 table_cls=RDDTable)
    else:
        raise EnvironmentError(f"{backend} unknown")


def init_federation(session_id, work_mode, runtime_conf, server_conf_path) -> Federation:
    if work_mode.is_standalone():
        from .standalone import FederationRuntime
        return FederationRuntime(session_id, runtime_conf)
    elif work_mode.is_cluster():
        from .cluster import FederationRuntime
        server_conf = file_utils.load_json_conf(server_conf_path)
        if CONF_KEY_SERVER not in server_conf:
            raise EnvironmentError("server_conf should contain key {}".format(CONF_KEY_SERVER))
        if CONF_KEY_FEDERATION not in server_conf.get(CONF_KEY_SERVER):
            raise EnvironmentError(
                "The {} should be a json file containing key: {}".format(server_conf_path, CONF_KEY_FEDERATION))
        host = server_conf.get(CONF_KEY_SERVER).get(CONF_KEY_FEDERATION).get("host")
        port = server_conf.get(CONF_KEY_SERVER).get(CONF_KEY_FEDERATION).get("port")
        return FederationRuntime(session_id, runtime_conf, host, port)
    else:
        raise EnvironmentError(f"{work_mode} unknown")


def init(session_id, backend, work_mode, runtime_conf, server_conf_path) -> Tuple[Federation, FederationWrapped]:
    """
    This method is required before get/remote called.
    :param session_id: current job_id, None is ok, uuid will be used.
    :param work_mode: work mode
    :param backend: backend
    :param runtime_conf:
    :param server_conf_path:
    runtime_conf should be a dict with
     1. key "local" maps to the current process' role and party_id.
     2. key "role" maps to a dict mapping from each role to all involving party_ids.
     {
        "local": {
            "role": "host",
            "party_id": 1000
        }
        "role": {
            "host": [999, 1000, 1001],
            "guest": [10002]
        }
     }

    """
    federation = init_federation(session_id=session_id,
                                 work_mode=work_mode,
                                 runtime_conf=runtime_conf,
                                 server_conf_path=server_conf_path)
    table_wrapper = init_table_wrapper(session_id=session_id,
                                       work_mode=work_mode,
                                       backend=backend)

    return federation, table_wrapper
