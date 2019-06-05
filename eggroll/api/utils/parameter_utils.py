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
from eggroll.api.utils.log_utils import getLogger

LOGGER = getLogger()


class ParameterOverride(object):
    @staticmethod
    def override_parameter(default_runtime_dict, setting_conf, submit_dict, out_prefix):

        if default_runtime_dict is None or submit_dict is None:
            raise Exception("default runtime conf and submit conf should be a json file")

        _method = submit_dict['task']
        _module = submit_dict['module']

        _module_setting = setting_conf['module'].get(_module)

        if not _module_setting:
            raise Exception("{} is not set in setting_conf ".format(_module))

        for role in submit_dict["role"]:
            _role_setting = _module_setting["role"].get(role)
            if not _role_setting:
                continue
            if _method not in _role_setting['tasklist']:
                continue
            _code_path = os.path.join(_module_setting.get('module_path'), _role_setting.get('program'))
            partyid_list = submit_dict["role"][role]
            for idx in range(len(partyid_list)):
                runtime_json = copy.deepcopy(default_runtime_dict)
                runtime_json['WorkFlowParam']['method'] = _method
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
                            if isinstance(valuelist, list):
                                if len(valuelist) <= idx:
                                    continue
                                else:
                                    runtime_json[param_class][attr] = valuelist[idx]
                            else:
                                runtime_json[param_class][attr] = valuelist
                runtime_json['local'] = submit_dict.get('local', {})
                my_local = {
                    "role": role, "party_id": partyid_list[idx]
                }
                runtime_json['local'].update(my_local)
                runtime_json['CodePath'] = _code_path
                runtime_json['module'] = _module
                output_path = os.path.join(out_prefix, _method, _module, str(role),
                                           str(partyid_list[idx]), "runtime_conf.json")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "w") as fout:
                    fout.write(json.dumps(runtime_json, indent=4))
