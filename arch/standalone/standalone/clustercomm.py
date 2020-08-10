#
#  Copyright 2019 The Eggroll Authors. All Rights Reserved.
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

from arch.standalone.standalone.eggroll import _DTable
from arch.standalone.standalone.eggroll import Standalone
from arch.standalone.utils import file_utils
from arch.standalone.utils.log_utils import getLogger
import asyncio
from arch.standalone import StoreType

OBJECT_STORAGE_NAME = "__clustercomm__"
STATUS_TABLE_NAME = "__status__"

CONF_KEY_TARGET = "clustercomm"
CONF_KEY_LOCAL = "local"


def init(job_id, runtime_conf):
    global LOGGER
    LOGGER = getLogger()
    if CONF_KEY_LOCAL not in runtime_conf:
        raise EnvironmentError("runtime_conf should be a dict containing key: {}".format(CONF_KEY_LOCAL))
    _party_id = runtime_conf.get(CONF_KEY_LOCAL).get('party_id')
    _role = runtime_conf.get(CONF_KEY_LOCAL).get("role")
    return ClusterCommRuntime(job_id, _party_id, _role, runtime_conf)


async def check_status_and_get_value(_table, _key):
    _value = _table.get(_key)
    while _value is None:
        await asyncio.sleep(0.1)
        _value = _table.get(_key)
    LOGGER.debug("[GET] Got {} type {}".format(_key, 'Table' if isinstance(_value, tuple) else 'Object'))
    return _value


def _get_meta_table(_name, _job_id):
    return Standalone.get_instance().table(_name, _job_id, partition=10)


class ClusterCommRuntime(object):
    instance = None

    @staticmethod
    def __remote__object_key(*args):
        return "-".join(["{}".format(arg) for arg in args])

    @staticmethod
    def get_instance():
        if ClusterCommRuntime.instance is None:
            raise EnvironmentError("clustercomm should be initialized before use")
        return ClusterCommRuntime.instance

    def __init__(self, job_id, party_id, role, runtime_conf, trans_conf="eggroll/conf/transfer_conf.json"):
        self.trans_conf = file_utils.load_json_conf(trans_conf)
        self.job_id = job_id
        self.party_id = party_id
        self.role = role
        self.runtime_conf = runtime_conf
        self._loop = asyncio.get_event_loop()
        ClusterCommRuntime.instance = self

    def __get_parties(self, role):
        return self.runtime_conf.get('role').get(role)

    def __check_authorization(self, name, is_send=True):
        algorithm, sub_name = name.split(".")
        auth_dict = self.trans_conf.get(algorithm)

        if auth_dict is None:
            raise ValueError("{} did not set in transfer_conf.json".format(algorithm))

        if auth_dict.get(sub_name) is None:
            raise ValueError("{} did not set under algorithm {} in transfer_conf.json".format(sub_name, algorithm))

        if is_send and auth_dict.get(sub_name).get('src') != self.role:
            raise ValueError("{} is not allow to send from {}".format(sub_name, self.role))
        elif not is_send and self.role not in auth_dict.get(sub_name).get('dst'):
            raise ValueError("{} is not allow to receive from {}".format(sub_name, self.role))
        return algorithm, sub_name

    def remote(self, obj, name: str, tag: str, role=None, idx=-1):
        algorithm, sub_name = self.__check_authorization(name)

        auth_dict = self.trans_conf.get(algorithm)

        if idx >= 0:
            if role is None:
                raise ValueError("{} cannot be None if idx specified".format(role))
            parties = {role: [self.__get_parties(role)[idx]]}
        elif role is not None:
            if role not in auth_dict.get(sub_name).get('dst'):
                raise ValueError("{} is not allowed to receive {}".format(role, name))
            parties = {role: self.__get_parties(role)}
        else:
            parties = {}
            for _role in auth_dict.get(sub_name).get('dst'):
                parties[_role] = self.__get_parties(_role)

        for _role, _partyIds in parties.items():
            for _partyId in _partyIds:
                _tagged_key = self.__remote__object_key(self.job_id, name, tag, self.role, self.party_id, _role,
                                                        _partyId)
                _status_table = _get_meta_table(STATUS_TABLE_NAME, self.job_id)
                if isinstance(obj, _DTable):
                    obj.set_gc_disable()
                    _status_table.put(_tagged_key, (obj._type, obj._name, obj._namespace, obj._partitions))
                else:
                    _table = _get_meta_table(OBJECT_STORAGE_NAME, self.job_id)
                    _table.put(_tagged_key, obj)
                    _status_table.put(_tagged_key, _tagged_key)
                LOGGER.debug("[REMOTE] Sent {}".format(_tagged_key))

    def get(self, name, tag, idx=-1):
        algorithm, sub_name = self.__check_authorization(name, is_send=False)

        auth_dict = self.trans_conf.get(algorithm)

        src_role = auth_dict.get(sub_name).get('src')

        src_party_ids = self.__get_parties(src_role)

        if 0 <= idx < len(src_party_ids):
            # idx is specified, return the remote object
            party_ids = [src_party_ids[idx]]
        else:
            # idx is not valid, return remote object list
            party_ids = src_party_ids

        _status_table = _get_meta_table(STATUS_TABLE_NAME, self.job_id)

        LOGGER.debug("[GET] {} {} getting remote object {} from {} {}".format(self.role, self.party_id, tag, src_role,
                                                                              party_ids))
        tasks = []

        for party_id in party_ids:
            _tagged_key = self.__remote__object_key(self.job_id, name, tag, src_role, party_id, self.role,
                                                    self.party_id)
            tasks.append(check_status_and_get_value(_status_table, _tagged_key))
        results = self._loop.run_until_complete(asyncio.gather(*tasks))

        rtn = []

        _object_table = _get_meta_table(OBJECT_STORAGE_NAME, self.job_id)
        for r in results:
            if isinstance(r, tuple):
                _persistent = r[0] == StoreType.LMDB
                rtn.append(
                    Standalone.get_instance().table(name=r[1], namespace=r[2], persistent=_persistent, partition=r[3]))
            else:
                rtn.append(_object_table.get(r))

        if 0 <= idx < len(src_party_ids):
            return rtn[0]
        return rtn
