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

import typing

from federatedml.framework.scheduler.setting_conf_parser import (
    SettingConf,
    get_all_module,
)


class ComponentMeta:
    def __init__(self, name) -> None:
        self.name = name

    def get_run_obj(self, role: str):
        return SettingConf.load(self.name).option_map_role(role, lambda p: p.get_obj())

    def get_run_obj_name(self, role: str) -> typing.Optional[str]:
        return SettingConf.load(self.name).option_map_role(role, lambda p: p.class_name)

    def get_param_obj(self, cpn_name: str):
        return (
            SettingConf.load(self.name)
            .param_program.get_obj()
            .set_name(f"{self.name}#{cpn_name}")
        )

    def get_supported_roles(self, roles=None):
        conf = SettingConf.load(self.name)
        if roles is None:
            return list(conf.roles)
        else:
            return [key for key in conf.roles if key in roles]


class Components:
    @classmethod
    def get_names(cls) -> typing.List[str]:
        return get_all_module()

    @classmethod
    def get(cls, name: str) -> ComponentMeta:
        return ComponentMeta(name)
