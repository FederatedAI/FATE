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
from federatedml.util.param_exact import ParamExtract


class ModelBase(object):
    def __init__(self):
        self.model_param = None
        self.data_dict = {}
        self.model_dict = {}
        pass

    def _init_runtime_parameters(self, component_parameters):
        param_extracter = ParamExtract()
        param_extracter.parse_param_from_config(self.model_param, component_parameters)

    def _init_model(self):
        pass

    def run(self, component_parameters={}, args={}):
        self._init_runtime_parameters(component_parameters)

        if "model" in args:
            self._init_model()

    def save_data(self):
        return self.data_dict

    def save_model(self):
        return self.model_dict