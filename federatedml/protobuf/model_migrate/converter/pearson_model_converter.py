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
from federatedml.protobuf.generated.pearson_model_param_pb2 import PearsonModelParam

from federatedml.protobuf.generated.pearson_model_meta_pb2 import PearsonModelMeta

from federatedml.protobuf.model_migrate.converter.converter_base import ProtoConverterBase
from typing import Dict
from federatedml.util import consts


class Replacer(object):
    def __init__(self, guest_mapping, host_mapping, arbiter_mapping):
        self._mapping = {
            consts.GUEST: guest_mapping,
            consts.HOST: host_mapping,
            consts.ARBITER: arbiter_mapping
        }

    def replace_tuple_format_party(self, tuple_str: str):
        role, party_id = tuple_str.strip("()").split(",")
        return f"({role}, {self._mapping[role][int(party_id)]})"

    def replace_anonymous_string(self, anonymous_str: str):
        role, party_id, index = anonymous_str.split("_")
        return f"{role}_{self._mapping[role][int(party_id)]}_{index}"

    def maybe_replace_anonymous_string(self, anonymous_str: str):
        try:
            role, party_id, index = anonymous_str.split("_")
        except Exception:
            return anonymous_str
        return f"{role}_{self._mapping[role][int(party_id)]}_{index}"


class HeteroPearsonConverter(ProtoConverterBase):

    def convert(self, param: PearsonModelParam, meta: PearsonModelMeta,
                guest_id_mapping: Dict,
                host_id_mapping: Dict,
                arbiter_id_mapping: Dict
                ):
        replacer = Replacer(guest_id_mapping, host_id_mapping, arbiter_id_mapping)
        param.party = replacer.replace_tuple_format_party(param.party)
        for i in range(len(param.parties)):
            param.parties[i] = replacer.replace_tuple_format_party(param.parties[i])
        for anonymous in param.anonymous_map:
            anonymous.anonymous = replacer.replace_anonymous_string(anonymous.anonymous)
        for names in param.all_names:
            for i, name in enumerate(names.names):
                names.names[i] = replacer.maybe_replace_anonymous_string(name)
        return param, meta
