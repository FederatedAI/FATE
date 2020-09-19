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

from federatedml.protobuf.generated.feature_selection_meta_pb2 import FeatureSelectionMeta
from federatedml.protobuf.generated.feature_selection_param_pb2 import FeatureSelectionParam, \
    FeatureSelectionFilterParam, FeatureValue, LeftCols
from federatedml.protobuf.model_migrate.converter.converter_base import AutoReplace
from federatedml.protobuf.model_migrate.converter.converter_base import ProtoConverterBase


class HeteroFeatureSelectionConverter(ProtoConverterBase):
    def convert(self, param: FeatureSelectionParam, meta: FeatureSelectionMeta,
                guest_id_mapping: Dict,
                host_id_mapping: Dict,
                arbiter_id_mapping: Dict
                ) -> Tuple:
        replacer = AutoReplace(guest_id_mapping, host_id_mapping, arbiter_id_mapping)

        host_col_name_objs = list(param.host_col_names)
        for col_obj in host_col_name_objs:
            old_party_id = col_obj.party_id
            col_obj.party_id = str(host_id_mapping[int(old_party_id)])
            col_names = list(col_obj.col_names)

            for idx, col_name in enumerate(col_names):
                col_obj.col_names[idx] = replacer.replace(col_name)

        filter_results = list(param.results)
        new_results = []
        for idx, result in enumerate(filter_results):
            host_feature_values = list(result.host_feature_values)
            new_feature_value_list = []
            for this_host in host_feature_values:
                feature_values = dict(this_host.feature_values)
                new_feature_values = {replacer.replace(k): v for k, v in feature_values.items()}
                new_feature_value_list.append(FeatureValue(feature_values=new_feature_values))

            left_col_list = list(result.host_left_cols)
            new_host_left_col = []
            for left_col_obj in left_col_list:
                original_cols = [replacer.replace(x) for x in left_col_obj.original_cols]
                left_cols = {replacer.replace(k): v for k, v in dict(left_col_obj.left_cols).items()}
                new_host_left_col.append(LeftCols(original_cols=original_cols,
                                                  left_cols=left_cols))
            new_result = FeatureSelectionFilterParam(feature_values=result.feature_values,
                                                     host_feature_values=new_feature_value_list,
                                                     left_cols=result.left_cols,
                                                     host_left_cols=new_host_left_col,
                                                     filter_name=result.filter_name)
            new_results.append(new_result)
        param = FeatureSelectionParam(
            results=new_results,
            final_left_cols=param.final_left_cols,
            col_names=param.col_names,
            host_col_names=param.host_col_names,
            header=param.header
        )
        return param, meta
