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

import builtins
import copy
import importlib
import json
import os

from fate_flow.utils.dsl_exception import *


class BaseParameterUtil(object):
    @staticmethod
    def override_parameter(**kwargs):
        pass

    @staticmethod
    def change_object_to_dict(obj):
        ret_dict = {}

        variable_dict = obj.__dict__
        for variable in variable_dict:
            attr = getattr(obj, variable)
            if attr and type(attr).__name__ not in dir(builtins):
                ret_dict[variable] = ParameterUtil.change_object_to_dict(attr)
            else:
                ret_dict[variable] = attr

        return ret_dict

    @staticmethod
    def _override_parameter(default_runtime_conf_prefix=None, setting_conf_prefix=None, submit_dict=None, module=None,
                            module_alias=None, version=1):

        _module_setting = ParameterUtil.get_setting_conf(setting_conf_prefix, module, module_alias)

        param_class_path = _module_setting["param_class"]
        param_class, param_obj = ParameterUtil.get_param_object(param_class_path, module, module_alias)

        default_runtime_dict = ParameterUtil.change_object_to_dict(param_obj)

        if version == 1:
            default_runtime_conf_suf = _module_setting["default_runtime_conf"]
            default_runtime_conf_buf = ParameterUtil.get_default_runtime_conf(default_runtime_conf_prefix,
                                                                              default_runtime_conf_suf,
                                                                              module,
                                                                              module_alias)

            default_runtime_dict = ParameterUtil.merge_parameters(default_runtime_dict, default_runtime_conf_buf,
                                                                  param_obj, component=module_alias, module=module,
                                                                  version=version)

        if not submit_dict:
            raise SubmitConfNotExistError()

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
                runtime_dict = {param_class: copy.deepcopy(default_runtime_dict)}
                for key, value in submit_dict.items():
                    if key not in ["algorithm_parameters", "role_parameters"]:
                        runtime_dict[key] = value

                if "algorithm_parameters" in submit_dict:
                    if module_alias in submit_dict["algorithm_parameters"]:
                        role_param_obj = copy.deepcopy(param_obj)
                        common_parameters = submit_dict["algorithm_parameters"].get(module_alias)
                        merge_dict = ParameterUtil.merge_parameters(runtime_dict[param_class],
                                                                    common_parameters,
                                                                    role_param_obj,
                                                                    component=module_alias,
                                                                    module=module,
                                                                    version=version)
                        runtime_dict[param_class] = merge_dict

                        try:
                            role_param_obj.check()
                        except Exception as e:
                            raise ParameterCheckError(component=module_alias, module=module, other_info=e)

                if "role_parameters" in submit_dict and role in submit_dict["role_parameters"]:
                    if version == 2:
                        role_parameters = submit_dict["role_parameters"][role]

                        role_idxs = role_parameters.keys()
                        for role_id in role_idxs:
                            role_param_obj = copy.deepcopy(param_obj)
                            if role_id == "all" or str(idx) in role_id.split("|"):
                                role_dict = role_parameters[role_id]
                                if module_alias in role_dict:
                                    parameters = role_dict.get(module_alias)
                                    merge_dict = ParameterUtil.merge_parameters(runtime_dict[param_class],
                                                                                parameters,
                                                                                role_param_obj,
                                                                                role_id,
                                                                                role,
                                                                                role_num=len(partyid_list),
                                                                                component=module_alias,
                                                                                module=module,
                                                                                version=version)

                                    runtime_dict[param_class] = merge_dict
                                    try:
                                        role_param_obj.check()
                                    except Exception as e:
                                        raise ParameterCheckError(component=module_alias, module=module, other_info=e)

                    if version == 1:
                        role_dict = submit_dict["role_parameters"][role]
                        if module_alias in role_dict:
                            role_param_obj = copy.deepcopy(param_obj)
                            role_parameters = role_dict.get(module_alias)
                            merge_dict = ParameterUtil.merge_parameters(runtime_dict[param_class],
                                                                        role_parameters,
                                                                        role_param_obj,
                                                                        idx,
                                                                        role,
                                                                        role_num=len(partyid_list),
                                                                        component=module_alias,
                                                                        module=module,
                                                                        version=version)
                            runtime_dict[param_class] = merge_dict

                            try:
                                role_param_obj.check()
                            except Exception as e:
                                raise ParameterCheckError(component=module_alias, module=module, other_info=e)

                runtime_dict['local'] = submit_dict.get('local', {})
                my_local = {
                    "role": role, "party_id": partyid_list[idx]
                }
                runtime_dict['local'].update(my_local)
                runtime_dict['CodePath'] = _code_path
                runtime_dict['module'] = module

                runtime_role_parameters[role].append(runtime_dict)

        return runtime_role_parameters

    @staticmethod
    def merge_parameters(runtime_dict, role_parameters, param_obj, idx=-1, role=None, role_num=0, component=None,
                         module=None, version=1):
        param_variables = param_obj.__dict__
        for key, val_list in role_parameters.items():
            if key not in param_variables:
                # continue
                raise RedundantParameterError(component=component, module=module, other_info=key)

            attr = getattr(param_obj, key)
            if type(attr).__name__ in dir(builtins) or not attr:
                if version == 1 and idx != -1:
                    if not isinstance(val_list, list):
                        raise RoleParameterNotListError(role=role, parameter=key)

                    if len(val_list) != role_num:
                        raise RoleParameterNotConsistencyError(role=role, parameter=key)

                    runtime_dict[key] = val_list[idx]
                    setattr(param_obj, key, val_list[idx])
                else:
                    runtime_dict[key] = val_list
                    setattr(param_obj, key, val_list)
            else:
                if key not in runtime_dict:
                    runtime_dict[key] = {}

                runtime_dict[key] = ParameterUtil.merge_parameters(runtime_dict.get(key),
                                                                   val_list,
                                                                   attr,
                                                                   idx,
                                                                   role=role,
                                                                   role_num=role_num,
                                                                   component=component,
                                                                   module=module,
                                                                   version=version)
                setattr(param_obj, key, attr)

        return runtime_dict

    @staticmethod
    def get_param_class_name(setting_conf_prefix, module):
        _module_setting_path = os.path.join(setting_conf_prefix, module + ".json")
        _module_setting = None
        with open(_module_setting_path, "r") as fin:
            _module_setting = json.loads(fin.read())

        param_class_path = _module_setting["param_class"]
        param_class = param_class_path.split("/", -1)[-1]

        return param_class

    @staticmethod
    def get_setting_conf(setting_conf_prefix, module, module_alias):
        _module_setting_path = os.path.join(setting_conf_prefix, module + ".json")
        if not os.path.isfile(_module_setting_path):
            raise ModuleNotExistError(component=module_alias, module=module)

        _module_setting = None
        fin = None
        try:
            fin = open(_module_setting_path, "r")
            _module_setting = json.loads(fin.read())
        except Exception as e:
            raise ModuleConfigError(component=module_alias, module=module, other_info=e)
        finally:
            if fin:
                fin.close()

        return _module_setting

    @staticmethod
    def get_param_object(param_class_path, module, module_alias):
        param_class = param_class_path.split("/", -1)[-1]

        param_module_path = ".".join(param_class_path.split("/", -1)[:-1]).replace(".py", "")
        if not importlib.util.find_spec(param_module_path):
            raise ParamClassNotExistError(component=module_alias, module=module,
                                          other_info="{} does not exist".format(param_module_path))

        param_module = importlib.import_module(param_module_path)

        if getattr(param_module, param_class) is None:
            raise ParamClassNotExistError(component=module_alias, module=module,
                                          other_info="{} does not exist is {}".format(param_class, param_module))

        param_obj = getattr(param_module, param_class)()

        return param_class, param_obj

    @staticmethod
    def get_default_runtime_conf(default_runtime_conf_prefix, default_runtime_conf_suf, module, module_alias):
        if not os.path.isfile(os.path.join(default_runtime_conf_prefix, default_runtime_conf_suf)):
            raise DefaultRuntimeConfNotExistError(component=module_alias, module=module)

        try:
            fin = open(os.path.join(default_runtime_conf_prefix, default_runtime_conf_suf))
            default_runtime_conf_buf = json.loads(fin.read())
        except Exception as e:
            raise DefaultRuntimeConfNotJsonError(component=module_alias, module=module, other_info=e)

        if default_runtime_conf_buf is None:
            raise DefaultRuntimeConfNotJsonError(component=module_alias, module=module)

        return default_runtime_conf_buf


class ParameterUtil(BaseParameterUtil):
    @staticmethod
    def override_parameter(default_runtime_conf_prefix=None, setting_conf_prefix=None, submit_dict=None, module=None,
                           module_alias=None):

        return ParameterUtil()._override_parameter(default_runtime_conf_prefix=default_runtime_conf_prefix,
                                                   setting_conf_prefix=setting_conf_prefix,
                                                   submit_dict=submit_dict,
                                                   module=module,
                                                   module_alias=module_alias,
                                                   version=1)

    @staticmethod
    def get_args_input(submit_dict, module="args"):
        if "role_parameters" not in submit_dict:
            return {}

        roles = submit_dict["role_parameters"].keys()
        if not roles:
            return {}

        args_input = {}
        args_datakey = set()

        for role in roles:
            if not submit_dict["role_parameters"][role].get(module):
                continue
            partyid_list = submit_dict["role"][role]

            args_parameters = submit_dict["role_parameters"][role].get(module)
            args_input[role] = []

            if "data" in args_parameters:
                dataset = args_parameters.get("data")
                for data_key in dataset:
                    datalist = dataset[data_key]

                    if len(datalist) != len(partyid_list):
                        raise RoleParameterNotConsistencyError(role=role, parameter=data_key)

                    args_datakey.add(data_key)

                    for i in range(len(datalist)):
                        value = datalist[i];
                        if len(args_input[role]) <= i:
                            args_input[role].append({module:
                                                         {"data":
                                                              {}
                                                          }
                                                     })

                        args_input[role][i][module]["data"][data_key] = value

        return args_input, args_datakey


class ParameterUtilV2(BaseParameterUtil):
    @staticmethod
    def override_parameter(setting_conf_prefix=None, submit_dict=None, module=None,
                           module_alias=None):
        return ParameterUtil._override_parameter(setting_conf_prefix=setting_conf_prefix,
                                                 submit_dict=submit_dict,
                                                 module=module,
                                                 module_alias=module_alias,
                                                 version=2)

    @staticmethod
    def get_input_parameters(submit_dict, module="args"):
        if "role_parameters" not in submit_dict:
            return {}, {}

        roles = submit_dict["role_parameters"].keys()
        if not roles:
            return {}, {}

        input_parameters = {}
        input_datakey = set()

        for role in roles:
            role_parameters = submit_dict["role_parameters"][role]
            input_parameters[role] = [{module: {"data": {}}} for i in range(len(submit_dict["role"][role]))]

            for idx in role_parameters.keys():
                parameters = role_parameters[idx]
                if module not in parameters:
                    continue

                dataset = parameters[module]
                for data_key, data_val in dataset.items():
                    input_datakey.add(data_key)

                    if idx == "all":
                        partyid_list = submit_dict["role"][role]
                        for i in range(len(partyid_list)):
                            input_parameters[role][i][module]["data"][data_key] = data_val
                    else:
                        input_parameters[role][int(idx)][module]["data"][data_key] = data_val

        return input_parameters, input_datakey
