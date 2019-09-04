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

# noinspection PyProtectedMember
from arch.api.cluster.eggroll import _DTable
from arch.api.cluster.federation import init
from arch.api.table.eggroll.wrapped_dtable import DTable


class FederationRuntime(object):

    def __init__(self, job_id, runtime_conf, server_conf_path):

        self._eggroll_federation = init(job_id, runtime_conf, server_conf_path)
        self.job_id = job_id

    def _get_all(self, name, tag):
        rtn = self._eggroll_federation.get(name=name, tag=tag, idx=-1)
        if len(rtn) > 0 and isinstance(rtn[0], _DTable):
            rtn = [DTable(dtable=tbl, job_id=self.job_id) for tbl in rtn]
        return rtn

    def _get_single(self, name, tag, idx):
        rtn = self._eggroll_federation.get(name=name, tag=tag, idx=idx)
        if isinstance(rtn, _DTable):
            rtn = DTable(dtable=rtn, job_id=self.job_id)
        return rtn

    # noinspection PyProtectedMember
    def _idx_in_range(self, name, idx) -> bool:
        if idx < 0:
            return False
        algorithm, sub_name = self._eggroll_federation._FederationRuntime__check_authorization(name, is_send=False)
        auth_dict = self._eggroll_federation.trans_conf.get(algorithm)
        src_role = auth_dict.get(sub_name).get('src')
        src_party_ids = self._eggroll_federation._FederationRuntime__get_parties(src_role)
        return 0 <= idx < len(src_party_ids)

    def get(self, name, tag, idx):
        if self._idx_in_range(name, idx):
            return self._get_single(name, tag, idx)
        else:
            return self._get_all(name, tag)

    # noinspection PyProtectedMember
    def remote(self, obj, name: str, tag: str, role=None, idx=-1):
        if isinstance(obj, DTable):
            return self._eggroll_federation.remote(obj=obj._dtable, name=name, tag=tag, role=role, idx=idx)

        return self._eggroll_federation.remote(obj=obj, name=name, tag=tag, role=role, idx=idx)
