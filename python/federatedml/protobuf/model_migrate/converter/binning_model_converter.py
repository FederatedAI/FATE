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
from federatedml.protobuf.generated.feature_binning_param_pb2 import FeatureBinningParam
from federatedml.protobuf.model_migrate.converter.converter_base import AutoReplace
from federatedml.protobuf.model_migrate.converter.converter_base import ProtoConverterBase


class FeatureBinningConverter(ProtoConverterBase):
    def convert(self, param: FeatureBinningParam, meta: FeatureBinningMeta,
                guest_id_mapping: Dict,
                host_id_mapping: Dict,
                arbiter_id_mapping: Dict
                ) -> Tuple:
        header_anonymous_list = list(param.header_anonymous)
        replacer = AutoReplace(guest_id_mapping, host_id_mapping, arbiter_id_mapping)

        for idx, h in enumerate(header_anonymous_list):
            param.header_anonymous[idx] = replacer.replace(h)
        return param, meta
