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

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class ComponentProperties(object):
    def __init__(self):
        self.need_cv = False
        self.need_run = False
        self.has_model = False
        self.has_isometric_model = False
        self.has_train_data = False
        self.has_eval_data = False
        self.has_normal_input_data = False
        self.role = None
        self.host_party_idlist = []
        self.local_partyid = -1

    def parse_component_param(self, component_parameters):
        try:
            need_cv = component_parameters.cv_param.need_cv
        except AttributeError:
            need_cv = False
        self.need_cv = need_cv
        try:
            need_run = component_parameters.need_run
        except AttributeError:
            need_run = True
        self.need_run = need_run
        LOGGER.debug("need_run: {}, need_cv: {}".format(self.need_run, self.need_cv))
        self.role = component_parameters["local"]["role"]
        self.host_party_idlist = component_parameters["role"]["host"]
        self.local_partyid = component_parameters["local"]["party_id"]
        return self

    def parse_dsl_args(self, args):
        if "model" in args:
            self.has_model = True
        if "isometric_model" in args:
            self.has_isometric_model = True
        data_sets = args["data"]
        for data_key in data_sets:
            if 'train_data' in data_sets[data_key]:
                self.has_train_data = True
            if 'eval_data' in data_sets[data_key]:
                self.has_eval_data = True
            if 'data' in data_sets[data_key]:
                self.has_normal_input_data = True
        return self

    @staticmethod
    def extract_input_data(args):
        data_sets = args["data"]
        train_data = None
        eval_data = None
        data = None

        for data_key in data_sets:
            if data_sets[data_key].get("train_data", None):
                train_data = data_sets[data_key]["train_data"]

            if data_sets[data_key].get("eval_data", None):
                eval_data = data_sets[data_key]["eval_data"]

            if data_sets[data_key].get("data", None):
                data = data_sets[data_key]["data"]
        return train_data, eval_data, data

    def extract_running_rules(self, args, model):
        train_data, eval_data, data = self.extract_input_data(args)

        todo_func_list = []
        todo_func_params = []

        if not self.need_run:
            todo_func_list.append(self.pass_data)
            todo_func_params.append([data])
            return todo_func_list, todo_func_params

        if self.need_cv:
            todo_func_list.append(model.cross_validation)
            todo_func_params.append([train_data])
            return todo_func_list, todo_func_params

        if self.has_model or self.has_isometric_model:
            todo_func_list.append(model.load_model)
            todo_func_params.append([args])

        if self.has_train_data:
            todo_func_list.extend([model.set_flowid, model.fit, model.set_flowid, model.predict])
            todo_func_params.extend([['fit'], [train_data], ['validate'], [train_data, 'validate']])

        if self.has_eval_data:
            todo_func_list.extend([model.set_flowid, model.predict])
            todo_func_params.extend([['predict'], [eval_data, 'predict']])

        if self.has_normal_input_data and not self.has_model:
            todo_func_list.extend([model.set_flowid, model.fit])
            todo_func_params.extend([['fit'], [data]])

        if self.has_normal_input_data and self.has_model:
            todo_func_list.extend([model.set_flowid, model.transform])
            todo_func_params.extend([['transform'], [data]])

        return todo_func_list, todo_func_params

    @staticmethod
    def pass_data(data):
        return data
