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


from typing import Dict, Tuple

from federatedml.protobuf.generated.feature_binning_meta_pb2 import FeatureBinningMeta
from federatedml.protobuf.generated.feature_binning_param_pb2 import FeatureBinningParam, IVParam
from federatedml.protobuf.model_migrate.converter.converter_base import AutoReplace
from federatedml.protobuf.model_migrate.converter.converter_base import ProtoConverterBase
from google.protobuf.json_format import MessageToDict


class FeatureBinningConverter(ProtoConverterBase):
    def convert(self, param: FeatureBinningParam, meta: FeatureBinningMeta,
                guest_id_mapping: Dict,
                host_id_mapping: Dict,
                arbiter_id_mapping: Dict
                ) -> Tuple:
        header_anonymous = list(param.header_anonymous)
        replacer = AutoReplace(guest_id_mapping, host_id_mapping, arbiter_id_mapping)

        param.header_anonymous[:] = replacer.migrate_anonymous_header(header_anonymous)

        self._migrate_binning_result(param, replacer, guest_id_mapping, host_id_mapping)

        if param.multi_class_result.host_party_ids:
            migrate_host_party_ids = []
            for host_party_id in param.multi_class_result.host_party_ids:
                migrate_host_party_ids.append(str(host_id_mapping[int(host_party_id)]))

            param.multi_class_result.host_party_ids[:] = migrate_host_party_ids

        self._migrate_binning_result(param.multi_class_result, replacer, guest_id_mapping, host_id_mapping, multi=True)

        return param, meta

    def _migrate_binning_result(self, param, replacer, guest_id_mapping, host_id_mapping, multi=False):
        if multi:
            for binning_result in param.results:
                migrate_party_id = self.migrate_binning_result(binning_result, guest_id_mapping, host_id_mapping)
                if migrate_party_id is not None:
                    binning_result.party_id = migrate_party_id
        else:
            migrate_party_id = self.migrate_binning_result(param.binning_result, guest_id_mapping, host_id_mapping)
            if migrate_party_id is not None:
                param.binning_result.party_id = migrate_party_id

        for host_binning_result in param.host_results:
            migrate_party_id = self.migrate_binning_result(host_binning_result, guest_id_mapping, host_id_mapping)
            if migrate_party_id is not None:
                host_binning_result.party_id = migrate_party_id

            kv_binning_result = dict(host_binning_result.binning_result)
            for col_name, iv_param in kv_binning_result.items():
                migrate_col_name = replacer.migrate_anonymous_header(col_name)
                host_binning_result.binning_result[migrate_col_name].CopyFrom(iv_param)
                del host_binning_result.binning_result[col_name]

    @staticmethod
    def migrate_binning_result(binning_result, guest_id_mapping, host_id_mapping):
        if binning_result.role and binning_result.party_id:
            party_id = int(binning_result.party_id)
            role = binning_result.role
            if role == "guest":
                migrate_party_id = guest_id_mapping[party_id]
            elif role == "host":
                migrate_party_id = host_id_mapping[party_id]
            else:
                raise ValueError(f"unsupported role {role} in binning migration")

            return str(migrate_party_id)

        return None
