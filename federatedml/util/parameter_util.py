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
import os
import copy


class ParameterOverride(object):
    @staticmethod
    def override_parameter(default_runtime_conf_prefix=None, setting_conf_prefix=None, submit_conf=None, module=None,
                           module_alias=None):

        default_runtime_dict = None
        with open(os.path.join(default_runtime_conf_prefix, module + "Param.json"), "r") as fin:
            default_runtime_dict = json.loads(fin.read())

        if default_runtime_dict is None:
            raise Exception("default runtime conf should be a json file")

        _module_setting_path = os.path.join(setting_conf_prefix, module + ".json")
        _module_setting = None
        with open(_module_setting_path, "r") as fin:
            _module_setting = json.loads(fin.read())

        if not _module_setting:
            raise Exception("{} is not set in setting_conf ".format(module))

        submit_dict = None
        with open(submit_conf, "r") as fin:
            submit_dict = json.loads(fin.read())

        if not submit_dict:
            raise ValueError("submit conf does exist or format is wrong")

        runtime_role_parameters = {}

        _support_rols = _module_setting["role"].keys()
        for role in submit_dict["role"]:
            _role_setting = None
            for _rolelist in _support_rols:
                if role not in _rolelist.split("|"):
                    continue
                else:
                    _role_setting = _module_setting["role"].get(_rolelist)

            if not _role_setting:
                continue

            _code_path = os.path.join(_module_setting.get('module_path'), _role_setting.get('program'))
            partyid_list = submit_dict["role"][role]
            runtime_role_parameters[role] = []

            for idx in range(len(partyid_list)):
                runtime_json = copy.deepcopy(default_runtime_dict)
                for key, value in submit_dict.items():
                    if key not in ["algorithm_parameters", "role_parameters"]:
                        runtime_json[key] = value

                if "algorithm_parameters" in submit_dict:
                    if module_alias in submit_dict["algorithm_parameters"]:
                        common_parameters = submit_dict["algorithm_parameters"].get(module_alias)
                        merge_json = ParameterOverride.merge_common_parameters(runtime_json[module + "Param"], common_parameters)
                        runtime_json = {module + "Param": merge_json}
                
                if "role_parameters" in submit_dict and role in submit_dict["role_parameters"]:
                    role_dict = submit_dict["role_parameters"][role]
                    if module_alias in role_dict:
                        role_parameters = role_dict.get(module_alias)
                        merge_json = ParameterOverride.merge_role_parameters(runtime_json[module + "Param"], role_parameters, idx)
                        runtime_json = {module + "Param": merge_json}
                
                runtime_json['local'] = submit_dict.get('local', {})
                my_local = {
                    "role": role, "party_id": partyid_list[idx]
                }
                runtime_json['local'].update(my_local)
                runtime_json['CodePath'] = _code_path
                runtime_json['module'] = module

                runtime_role_parameters[role].append(runtime_json)
        
        return runtime_role_parameters

    @staticmethod
    def merge_common_parameters(runtime_json, common_parameters):
        for key, val in common_parameters.items():
            if key not in runtime_json:
                runtime_json[key] = val
            elif type(val).__name__ == "dict":
                runtime_json[key] = ParameterOverride.merge_common_parameters(runtime_json.get(key), val)
            else:
                runtime_json[key] = val

        return runtime_json

    @staticmethod
    def merge_role_parameters(runtime_json, role_parameters, idx):
        for key, val_list in role_parameters.items():
            if len(val_list) < idx:
                continue

            val = val_list[idx]
            if key not in runtime_json:
                runtime_json[key] = val
            elif type(val).__name__ == "dict":
                runtime_json[key] = ParameterOverride.merge_common_parameters(runtime_json.get(key), val)
            else:
                runtime_json[key] = val

        return runtime_json

    @staticmethod
    def get_args_input(submit_conf, module="args"):
        submit_dict = None
        with open(submit_conf, "r") as fin:
            submit_dict = json.loads(fin.read())

        if "role_parameters" not in submit_dict:
            return {}

        roles = submit_dict["role_parameters"].keys()
        if not roles:
            return {}

        args_input = {}

        for role in roles:
            if not submit_dict["role_parameters"][role].get(module):
                continue

            args_parameters = submit_dict["role_parameters"][role].get(module)
            args_input[role] = []
  
            if "data" in args_parameters:
                dataset = args_parameters.get("data")
                for data_key in dataset:
                    datalist = dataset[data_key]
                    for i in range(len(datalist)):
                        value = datalist[i];
                        if len(args_input[role]) <= i:
                            args_input[role].append({module: 
                                                      {"data": 
                                                        {}
                                                      }
                                                    })

                        args_input[role][i][module]["data"][data_key] = value

        return args_input

                        
