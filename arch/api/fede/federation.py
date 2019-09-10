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


class FederationWrapped(object):

    # noinspection PyProtectedMember
    def __init__(self, job_id, runtime_conf, server_conf_path, work_mode, wrapper_table_cls):

        if work_mode.is_standalone():
            from .standalone import init
            from arch.api.standalone.eggroll import _DTable
            self._raw_federation = init(job_id, runtime_conf)
            self._raw_table_cls = _DTable
        elif work_mode.is_cluster():
            from .cluster import init
            from arch.api.cluster.eggroll import _DTable
            self._raw_federation = init(job_id, runtime_conf, server_conf_path)
            self._raw_table_cls = _DTable
        self.job_id = job_id
        self._wrapper_table_cls = wrapper_table_cls

    def _is_raw_table(self, obj):
        return isinstance(obj, self._raw_table_cls)

    def _is_wrapped_table(self, obj):
        return isinstance(obj, self._wrapper_table_cls)

    def _as_wrapped_table(self, obj):
        return self._wrapper_table_cls.from_dtable(dtable=obj, job_id=self.job_id)

    @staticmethod
    def _as_raw_table(obj: Table):
        return obj.dtable()

    def _get_all(self, name, tag):
        rtn = self._raw_federation.get(name=name, tag=tag, idx=-1)
        if len(rtn) > 0 and self._is_raw_table(rtn[0]):
            rtn = [self._as_wrapped_table(tbl) for tbl in rtn]
        return rtn

    def _get_single(self, name, tag, idx):
        rtn = self._raw_federation.get(name=name, tag=tag, idx=idx)
        if self._is_raw_table(rtn):
            rtn = self._as_wrapped_table(rtn)
        return rtn

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
        value = obj
        if self._is_wrapped_table(obj):
            value = self._as_raw_table(value)
        return self._raw_federation.remote(obj=value, name=name, tag=tag, role=role, idx=idx)


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
