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

import copy

from federatedml.protobuf.generated.feature_binning_param_pb2 import FeatureBinningParam


def extract_woe_array_dict(model_param_dict, host_idx=0):
    if len(model_param_dict.get("multiClassResults", {}).get("labels", [])) > 2:
        raise ValueError(f"Does not support transforming model trained on multi-label data. Please check.")
    host_result = model_param_dict.get("hostResults", [])[host_idx].get("binningResult", {})
    woe_array_dict = {}
    for col, res in host_result.items():
        woe_array_dict[col] = {"woeArray": res.get("woeArray", [])}

    return woe_array_dict


"""def merge_woe_array_dict(model_param_dict, woe_array_dict):
    header, anonymous_header = model_param_dict.get("header"), model_param_dict.get("headerAnonymous")
    if len(header) != len(anonymous_header):
        raise ValueError(f"Given header length and anonymous header length in model param do not match."
                         f"Please check!")
    anonymous_col_name_dict = dict(zip(header, anonymous_header))
    binning_results = copy.deepcopy(model_param_dict.get("binningResult", {}).get("binningResult", {}))

    for col_name, col_result in binning_results.items():
        if woe_array_dict.get(anonymous_col_name_dict[col_name]):
            col_result["woeArray"] = woe_array_dict[anonymous_col_name_dict[col_name]].get("woeArray")
    model_param_dict.get("binningResult", {})["binningResult"] = binning_results

    binning_results = model_param_dict.get("multiClassResult", {}).get("results", [])[0].get("binningResult")
    for col_name, col_result in binning_results.items():
        if woe_array_dict.get(anonymous_col_name_dict[col_name]):
            col_result["woeArray"] = woe_array_dict[anonymous_col_name_dict[col_name]].get("woeArray")
    # model_param_dict["multiClassResult"]["results"][0]["binningResult"] = binning_results
    return model_param_dict"""

def merge_woe_array_dict(model_param_pb, woe_array_dict):
    header, anonymous_header = list(model_param_pb.header), list(model_param_pb.header_anonymous)
    if len(header) != len(anonymous_header):
        raise ValueError(f"Given header length and anonymous header length in model param do not match."
                         f"Please check!")
    anonymous_col_name_dict = dict(zip(header, anonymous_header))
    binning_results = model_param_pb.binning_result.binning_result
    binning_results_cols = dict(binning_results).keys()

    for col_name in binning_results_cols:
        if woe_array_dict.get(anonymous_col_name_dict[col_name]):
            binning_results[col_name].woe_array = woe_array_dict[anonymous_col_name_dict[col_name]].get("woeArray")

    binning_results = model_param_pb.multi_class_result.results[0].binning_result
    binning_results_cols = dict(binning_results).keys()
    for col_name in binning_results_cols:
        if woe_array_dict.get(anonymous_col_name_dict[col_name]):
            binning_results[col_name].woe_array = woe_array_dict[anonymous_col_name_dict[col_name]].get("woeArray")
    # model_param_dict["multiClassResult"]["results"][0]["binningResult"] = binning_results
    return model_param_pb


def set_model_meta(model_meta_dict):
    model_meta_dict.get("transformParam", {})["transformType"] = "woe"
    return model_meta_dict
