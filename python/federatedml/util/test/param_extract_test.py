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
        self.config_dict = \
                {"BoostingTreeParam": {
                    "init_param": {"init_method": "test_init", "fit_intercept": False},
                    "tree_param": {"criterion_method": "test_decisiontree"},
                    "task_type": "test_boostingtree",
                    "test_variable": "test"}
                }

    def test_directly_extract(self):
        boosting_tree_param = BoostingTreeParam()
        extractor = ParamExtract()
        boosting_tree_param = extractor.parse_param_from_config(boosting_tree_param, self.config_dict)
        self.assertTrue(boosting_tree_param.task_type == "test_boostingtree")

    def test_undefine_variable_extract(self):
        boosting_tree_param = BoostingTreeParam()
        extractor = ParamExtract()
        boosting_tree_param = extractor.parse_param_from_config(boosting_tree_param, self.config_dict)
        self.assertTrue(not hasattr(boosting_tree_param, "test_variable"))

    def test_param_embedding(self):
        boosting_tree_param = BoostingTreeParam()
        extractor = ParamExtract()
        boosting_tree_param = extractor.parse_param_from_config(boosting_tree_param, self.config_dict)
        print ("boosting_tree_param.tree_param.criterion_method {}".format(boosting_tree_param.tree_param.criterion_method))
        self.assertTrue(boosting_tree_param.tree_param.criterion_method == "test_decisiontree")


if __name__ == '__main__':
    unittest.main()
