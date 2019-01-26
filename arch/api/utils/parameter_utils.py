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

import json


class ParameterOverride(object):
    @staticmethod
    def override_parameter(default_runtime_conf, submit_dict, out_prefix):
        default_runtime_dict = None

        with open(default_runtime_conf, "r") as fin:
            default_runtime_dict = json.loads(fin.read())

        if default_runtime_dict is None or submit_dict is None:
            raise Exception("default runtime conf and submit conf should be a json file")

        for role in submit_dict["role"]:
            partyid_list = submit_dict["role"][role]
            for idx in range(len(partyid_list)):
                runtime_json = default_runtime_dict.copy()
                for key, value in submit_dict.items():
                    if key not in ["algorithm_parameters", "role_parameters"]:
                        runtime_json[key] = value

                if "algorithm_parameters" in submit_dict:
                    for param_class in submit_dict["algorithm_parameters"]:
                        if param_class not in runtime_json:
                            runtime_json[param_class] = {}
                        for attr, value in submit_dict["algorithm_parameters"][param_class].items():
                            if attr not in runtime_json[param_class]:
                                runtime_json[param_class][attr] = {}
                            runtime_json[param_class][attr] = value

                if "role_parameters" in submit_dict and role in submit_dict["role_parameters"]:
                    role_dict = submit_dict["role_parameters"][role]
                    for param_class in role_dict:
                        if param_class not in runtime_json:
                            runtime_json[param_class] = {}
                        for attr, valuelist in role_dict[param_class].items():
                            if len(valuelist) <= idx:
                                continue
                            runtime_json[param_class][attr] = valuelist[idx]
                output_path = out_prefix + str(role) + "_" + str(partyid_list[idx]) + "_runtime_conf.json"
                with open(output_path, "w") as fout:
                    fout.write(json.dumps(runtime_json))
