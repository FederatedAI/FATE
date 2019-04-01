#!/usr/bin/env python    
# -*- coding: utf-8 -*- 

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
################################################################################
#
#
################################################################################

# =============================================================================
# Param Exact Class
# =============================================================================

from federatedml.util import consts
import json
import inspect


class ParamExtract(object):
    def __init__(self):
        pass

    @staticmethod
    def parse_param_from_config(param_var, config_file):
        config_json = None
        with open(config_file, "r") as fin:
            config_json = json.loads(fin.read())

        # print('In param extract, config_json is :{}'.format(config_json))

        if config_json is None:
            raise Exception("config file is not valid, please have a check!")

        from federatedml.param import param
        valid_classes = [class_info[0] for class_info in inspect.getmembers(param, inspect.isclass)]
        param_var = ParamExtract.recursive_parse_param_from_config(param_var, config_json, valid_classes,
                                                                   param_parse_depth=0)

        return param_var

    @staticmethod
    def recursive_parse_param_from_config(param, config_json, valid_classes, param_parse_depth):
        if param_parse_depth > consts.PARAM_MAXDEPTH:
            raise ValueError("Param define nesting too deep!!!, can not parse it")

        default_section = type(param).__name__
        inst_variables = param.__dict__

        for variable in inst_variables:
            attr = getattr(param, variable)

            if type(attr).__name__ in valid_classes:
                sub_params = ParamExtract.recursive_parse_param_from_config(attr, config_json, valid_classes,
                                                                            param_parse_depth + 1)
                setattr(param, variable, sub_params)
            else:
                if default_section in config_json and variable in config_json[default_section]:
                    option = config_json[default_section][variable]
                    setattr(param, variable, option)

        return param

    @staticmethod
    def is_class(var):
        return hasattr(var, "__class__")
