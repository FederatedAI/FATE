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

import inspect
import json
import numpy as np
import random
import unittest
import os
from federatedml.util.param_extract import ParamExtract
from federatedml.util import param_checker
from federatedml.param import param


class TestParameterChecker(unittest.TestCase):
    def setUp(self):
        home_dir = os.path.split(os.path.realpath(__file__))[0]
        self.config_path = home_dir + '/../../../workflow/conf/default_runtime_conf.json'
        validation_path = home_dir + '/../../../workflow/conf/param_validation.json'
        with open(validation_path, "r") as fin:
            self.validation_json = json.loads(fin.read())

        self.all_checker = param_checker.AllChecker(self.config_path)

        self.param_classes = [class_info[0] for class_info in inspect.getmembers(param, inspect.isclass)]

    def test_checker(self):

        self._check(param.DataIOParam, param_checker.DataIOParamChecker)
        self._check(param.EncryptParam, param_checker.EncryptParamChecker)
        self._check(param.EvaluateParam, param_checker.EvaluateParamChecker)
        self._check(param.ObjectiveParam, param_checker.ObjectiveParamChecker)
        self._check(param.PredictParam, param_checker.PredictParamChecker)
        self._check(param.WorkFlowParam, param_checker.WorkFlowParamChecker)
        self._check(param.InitParam, param_checker.InitParamChecker)
        self._check(param.EncodeParam, param_checker.EncodeParamChecker)
        self._check(param.IntersectParam, param_checker.IntersectParamChecker)
        self._check(param.LogisticParam, param_checker.LogisticParamChecker)
        self._check(param.DecisionTreeParam, param_checker.DecisionTreeParamChecker)
        self._check(param.BoostingTreeParam, param_checker.BoostingTreeParamChecker)
        self._check(param.FTLModelParam, param_checker.FTLModelParamChecker)
        self._check(param.LocalModelParam, param_checker.LocalModelParamChecker)
        self._check(param.FTLDataParam, param_checker.FTLDataParamChecker)
        self._check(param.FTLValidDataParam, param_checker.FTLValidDataParamChecker)
        # self._check(param.FeatureBinningParam, param_checker.FeatureBinningParamChecker)
        # self._check(param.FeatureSelectionParam, param_checker.FeatureSelectionParamChecker)

    def _check(self, Param, Checker):
        param_obj = Param()
        param_obj = ParamExtract.parse_param_from_config(param_obj, self.config_path)
        Checker.check_param(param_obj)

        self.all_checker.validate_restricted_param(param_obj, self.validation_json, self.param_classes) 


if __name__ == '__main__':
    unittest.main()
