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

import copy
import functools
from collections import Iterable

from federatedml.statistic import data_overview
from federatedml.statistic.data_overview import get_header
from federatedml.feature.imputer import Imputer
from federatedml.util import LOGGER


class FeatureImputation(object):
    def __init__(self, params):
        # self.area = params.area
        self.mode = params.mode
        self.missing_value_list = params.missing_value_list
        self.missing_fill_method = params.missing_fill_method
        self.missing_impute = params.missing_impute

        self.summary_obj = None

        self.model_param_name = 'FeatureImputationParam'
        self.model_meta_name = 'FeatureImputationMeta'

    def get_model_summary(self):
        pass

    def export_model(self, need_run):
        meta_obj = self._get_meta(need_run)
        param_obj = self._get_param()
        result = {
            self.model_meta_name: meta_obj,
            self.model_param_name: param_obj
        }
        return result

    def fit(self, data):
        pass

    def transform(self, data):
        pass

    def load_model(self, name, namespace):
        pass

    def save_model(self, name, namespace):
        pass

    def _get_param(self):
        pass

    def _get_meta(self, need_run):
        pass