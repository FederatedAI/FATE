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

import json
import os
import unittest

from federatedml.param.feature_binning_param import FeatureBinningParam
from federatedml.util.param_extract import ParamExtract

home_dir = os.path.split(os.path.realpath(__file__))[0]


class TestParamExtract(unittest.TestCase):
    def setUp(self):
        self.param = FeatureBinningParam()
        json_config_file = home_dir + '/param_feature_binning.json'
        self.config_path = json_config_file
        with open(json_config_file, 'r', encoding='utf-8') as load_f:
            role_config = json.load(load_f)
        self.config_json = role_config

    # def tearDown(self):
    #     os.system("rm -r " + self.config_path)

    def test_directly_extract(self):
        param_obj = FeatureBinningParam()
        extractor = ParamExtract()
        param_obj = extractor.parse_param_from_config(param_obj, self.config_json)
        self.assertTrue(param_obj.method == "quantile")
        self.assertTrue(param_obj.transform_param.transform_type == 'bin_num')


if __name__ == '__main__':
    unittest.main()
