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
from typing import Dict

from federatedml.protobuf.generated.pearson_model_meta_pb2 import PearsonModelMeta
from federatedml.protobuf.generated.pearson_model_param_pb2 import PearsonModelParam
from federatedml.protobuf.model_migrate.converter.converter_base import ProtoConverterBase, AutoReplace


class HeteroPearsonConverter(ProtoConverterBase):

    def convert(self, param: PearsonModelParam, meta: PearsonModelMeta,
                guest_id_mapping: Dict,
                host_id_mapping: Dict,
                arbiter_id_mapping: Dict
                ):
        replacer = AutoReplace(guest_id_mapping, host_id_mapping, arbiter_id_mapping)
        param.party = replacer.party_tuple_format(param.party)
        for i in range(len(param.parties)):
            param.parties[i] = replacer.party_tuple_format(param.parties[i])
        for anonymous in param.anonymous_map:
            anonymous.anonymous = replacer.anonymous_format(anonymous.anonymous)
        for names in param.all_names:
            for i, name in enumerate(names.names):
                names.names[i] = replacer.maybe_anonymous_format(name)
        return param, meta
