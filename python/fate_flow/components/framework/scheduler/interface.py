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
"""
federatedml framwork，4 interface：
module parameter checker
get component module support roles
get component module object
get component parameters
"""
import importlib
import json
import os
from fate_flow.components.param.param_extract import ParamExtract


home_dir = os.path.split(os.path.realpath(__file__))[0]
setting_conf_dir = os.path.join(home_dir, "..", "..", "conf", "setting_conf")


def has_module(module):
    setting_path = os.path.join(setting_conf_dir, module + ".json")
    return True if os.path.isfile(setting_path) else False


def get_support_role(module, roles=None):
    setting_path = os.path.join(setting_conf_dir, module + ".json")
    with open(setting_path, "r") as fin:
        setting = json.loads(fin.read())
        support_roles = setting["role"]

        if not roles:
            ret = set()
            for sp_role in support_roles:
                support_role_list = __parse_support_role(sp_role)
                ret |= set(support_role_list)

            return list(ret)
        else:
            ret = set()
            for sp_role in support_roles:
                support_role_list = __parse_support_role(sp_role)
                for role in support_role_list:
                    if role in roles:
                        ret.add(role)

            return list(ret)


def get_module(module, role):
    setting_path = os.path.join(setting_conf_dir, module + ".json")
    with open(setting_path, "r") as fin:
        setting = json.loads(fin.read())
        for support_role in setting["role"]:
            roles = __parse_support_role(support_role)
            if role not in roles:
                continue

            object_path = setting["module_path"] + "/" + setting["role"][support_role]["program"]
            import_path = ".".join(object_path.split("/", -1)[:-1]).replace(".py", "")
            object_name = object_path.split("/", -1)[-1]
            module_obj = getattr(importlib.import_module(import_path), object_name)()

            return module_obj

    return None


def get_module_name(module, role):
    setting_path = os.path.join(setting_conf_dir, module + ".json")
    with open(setting_path, "r") as fin:
        setting = json.loads(fin.read())
        for support_role in setting["role"]:
            if support_role.find("|") != -1:
                support_role_list = support_role.split("|", -1)
            else:
                support_role_list = [support_role]
            if role in support_role_list:
                object_path = setting["module_path"] + "/" + setting["role"][support_role]["program"]
                object_name = object_path.split("/", -1)[-1]
                return object_name

    return None


def get_module_param(module, alias):
    setting_path = os.path.join(setting_conf_dir, module + ".json")
    with open(setting_path, "r") as fin:
        setting = json.loads(fin.read())
        param_class_path = setting["param_class"]
        param_class = param_class_path.split("/", -1)[-1]
        param_module_path = ".".join(param_class_path.split("/", -1)[:-1]).replace(".py", "")
        if not importlib.util.find_spec(param_module_path):
            raise ValueError(f"component={alias}, module={module}, path={param_class_path} does not exist")

        param_module = importlib.import_module(param_module_path)
        if getattr(param_module, param_class) is None:
            raise ValueError(
                f"component={alias}, module={module}, param_class={param_class}" +
                f"does not exist in param module={param_module}")

        param_obj = getattr(param_module, param_class)()

        return param_obj


def update_param(param, conf, valid_check=True, module=None, cpn=None):
    return ParamExtract().parse_param_from_config(param,
                                                {"ComponentParam": conf},
                                                valid_check=valid_check,
                                                module=module,
                                                cpn=cpn)


def check_param(param):
    param.check()


def change_param_to_dict(param):
    return ParamExtract().change_param_to_dict(param)


# this interface only support for dsl v1
def get_not_builtin_types_for_dsl_v1(param):
    return ParamExtract().get_not_builtin_types(param)


def __parse_support_role(support_role):
    if support_role.find("|") != -1:
        support_role_list = support_role.split("|", -1)
        for idx in range(len(support_role_list)):
            support_role_list[idx] = support_role_list[idx].strip()

        return support_role_list
    else:
        return [support_role]

