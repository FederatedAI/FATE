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

import unittest
import os

from federatedml.util.param_extract import ParamExtract
from federatedml.param import InitParam
from federatedml.param import BoostingTreeParam


class TestParamExtract(unittest.TestCase):
    def setUp(self):
        self.init_param = InitParam()
        self.boosting_tree_param = BoostingTreeParam()
        import json
        import time
        config_dict = \
            {"InitParam": {"init_method": "test_init", "fit_intercept": False},
             "DecisionTreeParam": {"criterion_method": "test_decisiontree"},
             "BoostingTreeParam": {"task_type": "test_boostingtree"}}
        config_json = json.dumps(config_dict)
        timeid = int(time.time() * 1000)
        self.config_path = "param_config_test." + str(timeid)
        with open(self.config_path, "w") as fout:
            fout.write(config_json)

    def tearDown(self):
        os.system("rm -r " + self.config_path)

    def test_directly_extract(self):
        init_param = InitParam()
        extractor = ParamExtract()
        init_param = extractor.parse_param_from_config(init_param, self.config_path)
        self.assertTrue(init_param.init_method == "test_init")

    def test_param_embedding(self):
        boosting_tree_param = BoostingTreeParam()
        extractor = ParamExtract()
        boosting_tree_param = extractor.parse_param_from_config(boosting_tree_param, self.config_path)
        self.assertTrue(boosting_tree_param.tree_param.criterion_method == "test_decisiontree")
        self.assertTrue(boosting_tree_param.task_type == "test_boostingtree")


if __name__ == '__main__':
    unittest.main()
