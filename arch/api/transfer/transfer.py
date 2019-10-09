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

from arch.api.table.table import Table
from arch.api.utils.splitable import is_splitable_obj, split_remote, split_get, is_split_head, \
    split_table_tag, get_num_split


class FederationWrapped(object):

    # noinspection PyProtectedMember
    def __init__(self, job_id, runtime_conf, server_conf_path, work_mode, wrapper_table_cls):

        if work_mode.is_standalone():
            from .standalone import init
            from eggroll.api.standalone.eggroll import _DTable
            self._raw_federation = init(job_id, runtime_conf)
            self._raw_table_cls = _DTable
        elif work_mode.is_cluster():
            from .cluster import init
            from eggroll.api.cluster.eggroll import _DTable
            self._raw_federation = init(job_id, runtime_conf, server_conf_path)
            self._raw_table_cls = _DTable
        self.job_id = job_id
        self._wrapper_table_cls = wrapper_table_cls

    def _is_raw_table(self, obj):
        return isinstance(obj, self._raw_table_cls)

    def _is_wrapped_table(self, obj):
        return isinstance(obj, self._wrapper_table_cls)

    def _as_wrapped_table(self, obj):
        return self._wrapper_table_cls.from_dtable(dtable=obj, session_id=self.job_id)

    @staticmethod
    def _as_raw_table(obj: Table):
        return obj.dtable()

    def _get_all(self, name, tag):
        rtn = self._raw_federation.get(name=name, tag=tag, idx=-1)
        return [self._post_get(value, name, tag, idx) for idx, value in enumerate(rtn)]

    def _get_single(self, name, tag, idx):
        rtn = self._raw_federation.get(name=name, tag=tag, idx=idx)
        return self._post_get(rtn, name, tag, idx)

    def _post_get(self, rtn, name, tag, idx):
        if self._is_raw_table(rtn):
            return self._as_wrapped_table(rtn)

        if not is_split_head(rtn):
            return rtn

        num_split = get_num_split(rtn)
        splits = [self._raw_federation.get(name=name, tag=split_table_tag(tag, i), idx=idx) for i in range(num_split)]
        obj = split_get(splits)
        return obj

    # noinspection PyProtectedMember
    def _idx_in_range(self, name, idx) -> bool:
        if idx < 0:
            return False
        algorithm, sub_name = self._raw_federation._FederationRuntime__check_authorization(name, is_send=False)
        auth_dict = self._raw_federation.trans_conf.get(algorithm)
        src_role = auth_dict.get(sub_name).get('src')
        src_party_ids = self._raw_federation._FederationRuntime__get_parties(src_role)
        return 0 <= idx < len(src_party_ids)

    def get(self, name, tag, idx):
        if self._idx_in_range(name, idx):
            return self._get_single(name, tag, idx)
        else:
            return self._get_all(name, tag)

    def remote(self, obj, name: str, tag: str, role=None, idx=-1):
        if not is_splitable_obj(obj):
            if self._is_wrapped_table(obj):
                obj = self._as_raw_table(obj)
            return self._raw_federation.remote(obj=obj, name=name, tag=tag, role=role, idx=idx)

        # maybe split remote
        value = split_remote(obj)

        # num fragment is 1
        if not is_split_head(value[0]):
            return self._raw_federation.remote(obj=value[0], name=name, tag=tag, role=role, idx=idx)

        # num fragment > 1
        self._raw_federation.remote(value[0], name=name, tag=tag, role=role, idx=idx)
        for k, v in value[1]:
            self._raw_federation.remote(obj=v,
                                        name=name,
                                        tag=split_table_tag(tag, k),
                                        role=role,
                                        idx=idx)
        return


class FederationBuilder(object):

    @staticmethod
    def build_eggroll_backend(job_id,
                              runtime_conf,
                              work_mode,
                              server_conf_path=None):
        from arch.api.table.eggroll.table_impl import DTable
        return FederationWrapped(job_id=job_id,
                                 runtime_conf=runtime_conf,
                                 server_conf_path=server_conf_path,
                                 work_mode=work_mode,
                                 wrapper_table_cls=DTable)

    @staticmethod
    def build_spark_backend(job_id,
                            runtime_conf,
                            work_mode,
                            server_conf_path=None):
        from arch.api.table.pyspark.table_impl import RDDTable
        return FederationWrapped(job_id=job_id,
                                 runtime_conf=runtime_conf,
                                 server_conf_path=server_conf_path,
                                 work_mode=work_mode,
                                 wrapper_table_cls=RDDTable)
