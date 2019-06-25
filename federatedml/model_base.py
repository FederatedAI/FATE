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
from federatedml.util.param_extract import ParamExtract


class ModelBase(object):
    def __init__(self):
        self.model_output = None
        self.mode = None
        self.data_output = None
        self.model_param = None

    def _init_runtime_parameters(self, component_parameters):
        param_extracter = ParamExtract()
        param_extracter.parse_param_from_config(self.model_param, component_parameters)
        self._init_model(self.model_param)

    def _init_model(self, param):
        pass

    def _load_model(self, model_dict):
        pass

    def _run_data(self, data_sets=None, stage=None):
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

        if train_data:
            self.fit(train_data)
            self.data_output = self.predict(train_data)
            self.data_output = self.data_output.mapValues(lambda value: (value, "train"))

            if eval_data:
                eval_data_output = self.predict(eval_data)
                eval_data_outputput = eval_data_output.mapValues(lambda value: (value, "predict"))

                self.data_output.union(eval_data_output)

        elif eval_data:
            self.data_output = self.predict(eval_data)
            self.data_output = self.data_output.mapValues(lambda value: (value, "predict"))

        else:
            if stage == "fit":
                self.data_output = self.fit(data)
            else:
                self.data_output = self.transform(data)

    def run(self, component_parameters=None, args=None):
        self._init_runtime_parameters(component_parameters)

        stage = None
        if "model" in args:
            self._load_model(args["model"])
            stage = "transform"
        elif "isometric_model" in args:
            self._load_model(args["isometric_model"])
            stage = "fit"
        else:
            stage = "fit"

        if args.get("data", None) is None:
            return

        self._run_data(args["data"], stage)

    def predict(self, data_inst):
        return data_inst

    def fit(self, data_inst):
        pass

    def transform(self, data_inst):
        pass

    def save_data(self):
        return self.data_output

    def export_model(self):
        self.model_output = {"XXXMeta": "model_meta",
                             "XXXParam": "model_param"}

    def save_model(self):
        return self.export_model()
