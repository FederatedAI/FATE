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
from federatedml.protobuf import parse_pb_buffer


def extract_woe_array_dict(model_param_dict, host_idx=0):
    if len(model_param_dict.get("multiClassResults", {}).get("labels", [])) > 2:
        raise ValueError(f"Does not support transforming model trained on multi-label data. Please check.")
    host_result = model_param_dict.get("hostResults", [])[host_idx].get("binningResult", {})
    woe_array_dict = {}
    for col, res in host_result.items():
        woe_array_dict[col] = {"woeArray": res.get("woeArray", [])}

    return woe_array_dict


def merge_woe_array_dict(pb_name, model_param_pb, model_param_dict, woe_array_dict):
    model_param_pb = parse_pb_buffer(pb_name, model_param_pb)

    header, anonymous_header = list(model_param_pb.header), list(model_param_pb.header_anonymous)
    if len(header) != len(anonymous_header):
        raise ValueError(
            "Given header length and anonymous header length in model param do not match. "
            "Please check!"
        )

    anonymous_col_name_dict = dict(zip(header, anonymous_header))

    for col_name in model_param_pb.binning_result.binning_result:
        try:
            woe_array = woe_array_dict[anonymous_col_name_dict[col_name]]["woeArray"]
        except KeyError:
            continue

        model_param_pb.binning_result.binning_result[col_name].woe_array[:] = woe_array
        model_param_dict["binningResult"]["binningResult"][col_name]["woeArray"] = woe_array

    for col_name in model_param_pb.multi_class_result.results[0].binning_result:
        try:
            woe_array = woe_array_dict[anonymous_col_name_dict[col_name]]["woeArray"]
        except KeyError:
            continue

        model_param_pb.multi_class_result.results[0].binning_result[col_name].woe_array[:] = woe_array
        model_param_dict["multiClassResult"]["results"][0]["binningResult"][col_name]["woeArray"] = woe_array

    return model_param_pb.SerializeToString(), model_param_dict


def set_model_meta(model_meta_dict):
    model_meta_dict.get("transformParam", {})["transformType"] = "woe"
    return model_meta_dict
