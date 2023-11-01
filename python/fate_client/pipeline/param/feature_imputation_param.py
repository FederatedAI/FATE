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

from pipeline.param import consts
from pipeline.param.base_param import BaseParam


class FeatureImputationParam(BaseParam):
    """
    Define feature imputation parameters

    Parameters
    ----------

    default_value : None or single object type
        the value to replace missing value.
        if None, it will use default value defined in federatedml/feature/imputer.py,
        if single object, will fill missing value with this object
    col_default_value: None or dict of (column name, default_value) pairs
        specifies default value for each column;
        any column to be filled with 'designated' but not specified in col_default_value will take default_value
    missing_fill_method : [None, 'min', 'max', 'mean', 'designated', 'median', 'mode']
        the method to replace missing value
    col_missing_fill_method: None or dict of (column name, missing_fill_method) pairs
        specifies method to replace missing value for each column;
        any column not specified will take missing_fill_method,
        if missing_fill_method is None, unspecified column will not be imputed;
    missing_impute : None or list
        element of list can be any type, or auto generated if value is None, define which values to be considered as missing, default: None
    error: float, 0 <= error < 1 default: 0.0001
        The error of tolerance for quantile calculation when computing median
    multi_mode: string, choose between 'random' and 'raise', default 'random'
        if 'random', randomly choose one of the modes as fill value, if 'raise', raise error when there are multiple modes
    need_run: bool, default True
        need run or not

    """

    def __init__(self, default_value=0, col_default_value=None, missing_fill_method=None, col_missing_fill_method=None,
                 missing_impute=None, need_run=True, error=consts.DEFAULT_RELATIVE_ERROR, multi_mode='random'):
        super(FeatureImputationParam, self).__init__()
        self.default_value = default_value
        self.col_default_value = col_default_value
        self.missing_fill_method = missing_fill_method
        self.col_missing_fill_method = col_missing_fill_method
        self.missing_impute = missing_impute
        self.need_run = need_run
        self.error = error
        self.multi_mode = multi_mode

    def check(self):

        descr = "feature imputation param's "

        self.check_boolean(self.need_run, descr + "need_run")

        if self.missing_fill_method is not None:
            self.missing_fill_method = self.check_and_change_lower(self.missing_fill_method,
                                                                   ['min', 'max', 'mean',
                                                                    'designated', 'median', 'mode'],
                                                                   f"{descr}missing_fill_method ")
        if self.col_default_value:
            if not isinstance(self.col_default_value, dict):
                raise ValueError(f"{descr}col_default_value should be a dict")
            for k, v in self.col_default_value.items():
                if not isinstance(k, str):
                    raise ValueError(f"{descr}col_default_value should contain str key(s) only")
        if self.col_missing_fill_method:
            if not isinstance(self.col_missing_fill_method, dict):
                raise ValueError(f"{descr}col_missing_fill_method should be a dict")
            for k, v in self.col_missing_fill_method.items():
                if not isinstance(k, str):
                    raise ValueError(f"{descr}col_missing_fill_method should contain str key(s) only")
                v = self.check_and_change_lower(v,
                                                ['min', 'max', 'mean', 'designated', 'median', 'mode'],
                                                f"per column method specified in {descr} col_missing_fill_method dict")
                self.col_missing_fill_method[k] = v
        if self.missing_impute:
            if not isinstance(self.missing_impute, list):
                raise ValueError(f"{descr}missing_impute must be None or list.")

        return True
