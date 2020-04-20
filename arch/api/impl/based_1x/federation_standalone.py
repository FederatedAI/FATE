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
import asyncio
from typing import Union, Tuple

from arch.api.base.federation import Rubbish, Party, Federation
from arch.api.base.utils.store_type import StoreTypes
from arch.api.utils.log_utils import getLogger
from eggroll.api.standalone.eggroll import Standalone
# noinspection PyProtectedMember
from eggroll.api.standalone.eggroll import _DTable

OBJECT_STORAGE_NAME = "__federation__"
STATUS_TABLE_NAME = "__status__"

LOGGER = getLogger()


async def check_status_and_get_value(_table, _key):
    _value = _table.get(_key)
    while _value is None:
        await asyncio.sleep(0.1)
        _value = _table.get(_key)
    LOGGER.debug("[GET] Got {} type {}".format(_key, 'Table' if isinstance(_value, tuple) else 'Object'))
    return _value


def _get_meta_table(_name, _job_id):
    return Standalone.get_instance().table(_name, _job_id, partition=10)


class FederationRuntime(Federation):

    @staticmethod
    def __remote__object_key(*args):
        return "-".join(["{}".format(arg) for arg in args])

    def __init__(self, session_id, runtime_conf):
        super().__init__(session_id, runtime_conf)

        self._loop = asyncio.get_event_loop()

    def remote(self, obj, name: str, tag: str, parties: Union[Party, list]) -> Rubbish:
        if isinstance(parties, Party):
            parties = [parties]
        self._remote_side_auth(name=name, parties=parties)

        rubbish = Rubbish(name, tag)
        for party in parties:
            _tagged_key = self.__remote__object_key(self._session_id, name, tag, self._role, self._party_id, party.role,
                                                    party.party_id)
            _status_table = _get_meta_table(STATUS_TABLE_NAME, self._session_id)
            if isinstance(obj, _DTable):
                obj.set_gc_disable()
                # noinspection PyProtectedMember
                _status_table.put(_tagged_key, (obj._type, obj._name, obj._namespace, obj._partitions))
                rubbish.add_table(obj)
                rubbish.add_obj(_status_table, _tagged_key)
            else:
                _table = _get_meta_table(OBJECT_STORAGE_NAME, self._session_id)
                _table.put(_tagged_key, obj)
                _status_table.put(_tagged_key, _tagged_key)
                rubbish.add_obj(_table, _tagged_key)
                rubbish.add_obj(_status_table, _tagged_key)
            LOGGER.debug("[REMOTE] Sent {}".format(_tagged_key))
        return rubbish

    def get(self, name: str, tag: str, parties: Union[Party, list]) -> Tuple[list, Rubbish]:
        if isinstance(parties, Party):
            parties = [parties]
        self._get_side_auth(name=name, parties=parties)

        _status_table = _get_meta_table(STATUS_TABLE_NAME, self._session_id)
        LOGGER.debug(f"[GET] {self._local_party} getting {name}.{tag} from {parties}")
        tasks = []

        for party in parties:
            _tagged_key = self.__remote__object_key(self._session_id, name, tag, party.role, party.party_id, self._role,
                                                    self._party_id)
            tasks.append(check_status_and_get_value(_status_table, _tagged_key))
        results = self._loop.run_until_complete(asyncio.gather(*tasks))
        rtn = []
        rubbish = Rubbish(name, tag)
        _object_table = _get_meta_table(OBJECT_STORAGE_NAME, self._session_id)
        for r in results:
            LOGGER.debug(f"[GET] {self._local_party} getting {r} from {parties}")
            if isinstance(r, tuple):
                _persistent = r[0] == StoreTypes.ROLLPAIR_LMDB
                table = Standalone.get_instance().table(name=r[1], namespace=r[2], persistent=_persistent,
                                                        partition=r[3])
                rtn.append(table)
                rubbish.add_table(table)

            else:  # todo: should standalone mode split large object?
                obj = _object_table.get(r)
                rtn.append(obj)
                rubbish.add_obj(_object_table, r)
                rubbish.add_obj(_status_table, r)
        return rtn, rubbish
