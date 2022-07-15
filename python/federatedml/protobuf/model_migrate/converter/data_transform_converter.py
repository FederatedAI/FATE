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

from federatedml.protobuf.generated.data_transform_meta_pb2 import DataTransformMeta
from federatedml.protobuf.generated.data_transform_param_pb2 import DataTransformParam
from federatedml.protobuf.model_migrate.converter.converter_base import AutoReplace
from federatedml.protobuf.model_migrate.converter.converter_base import ProtoConverterBase


class DataTransformConverter(ProtoConverterBase):
    def convert(self, param: DataTransformParam, meta: DataTransformMeta,
                guest_id_mapping: Dict,
                host_id_mapping: Dict,
                arbiter_id_mapping: Dict
                ) -> Tuple:
        try:
            anonymous_header = list(param.anonymous_header)
            replacer = AutoReplace(guest_id_mapping, host_id_mapping, arbiter_id_mapping)

            param.anonymous_header[:] = replacer.migrate_anonymous_header(anonymous_header)
            return param, meta
        except AttributeError:
            return param, meta
