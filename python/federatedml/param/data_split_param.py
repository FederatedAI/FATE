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

from federatedml.param.base_param import BaseParam
from federatedml.util import LOGGER


class DataSplitParam(BaseParam):
    """
    Define data split param that used in data split.

    Parameters
    ----------
    random_state : None or int, default: None
        Specify the random state for shuffle.

    test_size : float or int or None, default: 0.0
        Specify test data set size.
        float value specifies fraction of input data set, int value specifies exact number of data instances

    train_size : float or int or None, default: 0.8
        Specify train data set size.
        float value specifies fraction of input data set, int value specifies exact number of data instances

    validate_size : float or int or None, default: 0.2
        Specify validate data set size.
        float value specifies fraction of input data set, int value specifies exact number of data instances

    stratified : bool, default: False
        Define whether sampling should be stratified, according to label value.

    shuffle : bool, default: True
        Define whether do shuffle before splitting or not.

    split_points : None or list, default : None
        Specify the point(s) by which continuous label values are bucketed into bins for stratified split.
        eg.[0.2] for two bins or [0.1, 1, 3] for 4 bins

    need_run: bool, default: True
        Specify whether to run data split

    """

    def __init__(self, random_state=None, test_size=None, train_size=None, validate_size=None, stratified=False,
                 shuffle=True, split_points=None, need_run=True):
        super(DataSplitParam, self).__init__()
        self.random_state = random_state
        self.test_size = test_size
        self.train_size = train_size
        self.validate_size = validate_size
        self.stratified = stratified
        self.shuffle = shuffle
        self.split_points = split_points
        self.need_run = need_run

    def check(self):
        model_param_descr = "data split param's "
        if self.random_state is not None:
            if not isinstance(self.random_state, int):
                raise ValueError(f"{model_param_descr} random state should be int type")
            BaseParam.check_nonnegative_number(self.random_state, f"{model_param_descr} random_state ")

        if self.test_size is not None:
            BaseParam.check_nonnegative_number(self.test_size, f"{model_param_descr} test_size ")
            if isinstance(self.test_size, float):
                BaseParam.check_decimal_float(self.test_size, f"{model_param_descr} test_size ")
        if self.train_size is not None:
            BaseParam.check_nonnegative_number(self.train_size, f"{model_param_descr} train_size ")
            if isinstance(self.train_size, float):
                BaseParam.check_decimal_float(self.train_size, f"{model_param_descr} train_size ")
        if self.validate_size is not None:
            BaseParam.check_nonnegative_number(self.validate_size, f"{model_param_descr} validate_size ")
            if isinstance(self.validate_size, float):
                BaseParam.check_decimal_float(self.validate_size, f"{model_param_descr} validate_size ")
        # use default size values if none given
        if self.test_size is None and self.train_size is None and self.validate_size is None:
            self.test_size = 0.0
            self.train_size = 0.8
            self.validate_size = 0.2

        BaseParam.check_boolean(self.stratified, f"{model_param_descr} stratified ")
        BaseParam.check_boolean(self.shuffle, f"{model_param_descr} shuffle ")
        BaseParam.check_boolean(self.need_run, f"{model_param_descr} need run ")

        if self.split_points is not None:
            if not isinstance(self.split_points, list):
                raise ValueError(f"{model_param_descr} split_points should be list type")

        LOGGER.debug("Finish data_split parameter check!")
        return True
