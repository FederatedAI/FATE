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

import inspect
import unittest

from federatedml.param.ftl_param import FTLDataParam, FTLModelParam, LocalModelParam
from federatedml.util.param_extract import ParamExtract
# from federatedml.util.param_checker import FTLDataParamChecker, LocalModelParamChecker, FTLModelParamChecker

from arch.api.session import init


def get_filled_param(param_var, config_json):
    from federatedml.param import ftl_param
    valid_classes = [class_info[0] for class_info in inspect.getmembers(ftl_param, inspect.isclass)]
    param_var = ParamExtract.recursive_parse_param_from_config(param_var, config_json,
                                                               param_parse_depth=0)
    return param_var

"""
class TestParamValidation(unittest.TestCase):

    def test_correct_model_param_validation_test(self):
        correct_model_param = {
            "FTLModelParam": {
                "eps": 10e-3,
                "alpha": 100,
                "max_iter": 6,
                "is_encrypt": False
            }
        }
        ftl_model_param = FTLModelParam()
        ftl_model_param = get_filled_param(ftl_model_param, correct_model_param)
        FTLModelParamChecker.check_param(ftl_model_param)

    def test_model_param_incorrect_eps_validation_test(self):
        incorrect_eps = {
            "FTLModelParam": {
                "eps": -1,
                "alpha": 100,
                "max_iter": 10,
                "is_encrypt": False
            }
        }
        self.assertFTLModelParamValueError(incorrect_eps, " eps ")

    def test_model_param_incorrect_alpha_validation_test(self):
        incorrect_alpha = {
            "FTLModelParam": {
                "eps": 10e-3,
                "alpha": -1,
                "max_iter": 10,
                "is_encrypt": False
            }
        }
        self.assertFTLModelParamValueError(incorrect_alpha, " alpha ")

    def test_model_param_incorrect_maxiter_validation_test(self):
        incorrect_max_iter = {
            "FTLModelParam": {
                "eps": 10e-3,
                "alpha": 100,
                "max_iter": 0.6,
                "is_encrypt": False
            }
        }
        self.assertFTLModelParamValueError(incorrect_max_iter, " max_iter ")

    def test_model_param_incorrect_isencrypt_validation_test(self):
        incorrect_is_encrypt = {
            "FTLModelParam": {
                "eps": 10e-3,
                "alpha": 100,
                "max_iter": 6,
                "is_encrypt": 123
            }
        }
        self.assertFTLModelParamValueError(incorrect_is_encrypt, " is_encrypt ")

    def assertFTLModelParamValueError(self, param_json, param_to_validate):
        ftl_model_param = FTLModelParam()
        ftl_model_param = get_filled_param(ftl_model_param, param_json)
        with self.assertRaisesRegex(ValueError, param_to_validate):
            FTLModelParamChecker.check_param(ftl_model_param)

    def test_correct_local_model_param_validation_test(self):
        correct_local_model_param = {
            "LocalModelParam": {
                "encode_dim": 32,
                "learning_rate": 0.01
            }
        }
        ftl_local_model_param = LocalModelParam()
        ftl_local_model_param = get_filled_param(ftl_local_model_param, correct_local_model_param)
        LocalModelParamChecker.check_param(ftl_local_model_param)

    def test_model_param_incorrect_encode_dim_validation_test(self):
        incorrect_encode_dim = {
            "LocalModelParam": {
                "encode_dim": 0.9,
                "learning_rate": 0.01
            }
        }
        self.assertFTLLocalModelParamValueError(incorrect_encode_dim, " encode_dim ")

    def test_model_param_incorrect_learning_rate_validation_test(self):
        incorrect_learning_rate = {
            "LocalModelParam": {
                "encode_dim": 32,
                "learning_rate": 2
            }
        }
        self.assertFTLLocalModelParamValueError(incorrect_learning_rate, " learning_rate ")

    def assertFTLLocalModelParamValueError(self, param_json, param_to_validate):
        ftl_local_model_param = LocalModelParam()
        ftl_local_model_param = get_filled_param(ftl_local_model_param, param_json)
        with self.assertRaisesRegex(ValueError, param_to_validate):
            LocalModelParamChecker.check_param(ftl_local_model_param)

    def test_correct_data_param_validation_test(self):
        correct_data_param = {
            "FTLDataParam": {
                "file_path": "../data/UCI_Credit_Card.csv",
                "n_feature_guest": 10,
                "n_feature_host": 23,
                "overlap_ratio": 0.1,
                "guest_split_ratio": 0.9,
                "num_samples": 500,
                "balanced": True,
                "is_read_table": False
            }
        }
        ftl_data_param = FTLDataParam()
        ftl_data_param = get_filled_param(ftl_data_param, correct_data_param)
        FTLDataParamChecker.check_param(ftl_data_param)

    def test_model_param_incorrect_filepath_validation_test(self):
        incorrect_file_path = {
            "FTLDataParam": {
                "file_path": 1,
                "n_feature_guest": 10,
                "n_feature_host": 23,
                "overlap_ratio": 0.1,
                "guest_split_ratio": 0.9,
                "num_samples": 500,
                "balanced": True,
                "is_read_table": False
            }
        }
        self.assertFTLDataParamValueError(incorrect_file_path, " file_path ")

    def test_model_param_incorrect_nfeatureguest_validation_test(self):
        incorrect_n_feature_guest = {
            "FTLDataParam": {
                "file_path": "",
                "n_feature_guest": 1.2,
                "n_feature_host": 23,
                "overlap_ratio": 0.1,
                "guest_split_ratio": 0.9,
                "num_samples": 500,
                "balanced": True,
                "is_read_table": False
            }
        }
        self.assertFTLDataParamValueError(incorrect_n_feature_guest, " n_feature_guest ")

    def test_model_param_incorrect_nfeaturehost_validation_test(self):
        incorrect_n_feature_host = {
            "FTLDataParam": {
                "file_path": "",
                "n_feature_guest": 10,
                "n_feature_host": 2.2,
                "overlap_ratio": 0.1,
                "guest_split_ratio": 0.9,
                "num_samples": 500,
                "balanced": True,
                "is_read_table": False
            }
        }
        self.assertFTLDataParamValueError(incorrect_n_feature_host, " n_feature_host ")

    def test_model_param_incorrect_overlapratio_validation_test(self):
        incorrect_overlap_ratio = {
            "FTLDataParam": {
                "file_path": "",
                "n_feature_guest": 10,
                "n_feature_host": 23,
                "overlap_ratio": 10,
                "guest_split_ratio": 0.9,
                "num_samples": 500,
                "balanced": True,
                "is_read_table": False
            }
        }
        self.assertFTLDataParamValueError(incorrect_overlap_ratio, " overlap_ratio ")

    def test_model_param_incorrect_guestsplitratio_validation_test(self):
        incorrect_guest_split_ratio = {
            "FTLDataParam": {
                "file_path": "",
                "n_feature_guest": 10,
                "n_feature_host": 23,
                "overlap_ratio": 0.1,
                "guest_split_ratio": 9,
                "num_samples": 500,
                "balanced": True,
                "is_read_table": False
            }
        }
        self.assertFTLDataParamValueError(incorrect_guest_split_ratio, " guest_split_ratio ")

    def test_model_param_incorrect_numsamples_validation_test(self):
        incorrect_num_samples = {
            "FTLDataParam": {
                "file_path": "",
                "n_feature_guest": 10,
                "n_feature_host": 23,
                "overlap_ratio": 0.1,
                "guest_split_ratio": 0.9,
                "num_samples": 0.1,
                "balanced": True,
                "is_read_table": False
            }
        }
        self.assertFTLDataParamValueError(incorrect_num_samples, " num_samples ")

    def test_model_param_incorrect_balanced_validation_test(self):
        incorrect_balanced = {
            "FTLDataParam": {
                "file_path": "",
                "n_feature_guest": 10,
                "n_feature_host": 23,
                "overlap_ratio": 0.1,
                "guest_split_ratio": 0.9,
                "num_samples": 500,
                "balanced": 123,
                "is_read_table": False
            }
        }
        self.assertFTLDataParamValueError(incorrect_balanced, " balanced ")

    def test_model_param_incorrect_is_read_table_validation_test(self):
        incorrect_is_read_table = {
            "FTLDataParam": {
                "file_path": "",
                "n_feature_guest": 10,
                "n_feature_host": 23,
                "overlap_ratio": 0.1,
                "guest_split_ratio": 0.9,
                "num_samples": 500,
                "balanced": True,
                "is_read_table": 123
            }
        }
        self.assertFTLDataParamValueError(incorrect_is_read_table, " is_read_table ")

    def assertFTLDataParamValueError(self, param_json, param_to_validate):
        ftl_data_param = FTLDataParam()
        ftl_data_param = get_filled_param(ftl_data_param, param_json)
        with self.assertRaisesRegex(ValueError, param_to_validate):
            FTLDataParamChecker.check_param(ftl_data_param)
"""

if __name__=='__main__':
    init()
    unittest.main()
