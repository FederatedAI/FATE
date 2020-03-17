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
from arch.api.utils import log_utils
from federatedml.param.base_param import BaseParam

LOGGER = log_utils.getLogger()


class DataSplitParam(BaseParam):
    """
    Define data split param that used in data split.

    Parameters
    ----------
    random_state : None, int, default: None
        Specify the random state for shuffle.

    test_size : None, float, int, default: None
        Specify test data set size.
        float value specifies fraction of input data set, int value specifies exact number of data instances

    train_size : None, float, int, default: None
        Specify train data set size.
        float value specifies fraction of input data set, int value specifies exact number of data instances

    validate_size : None, float, int, default: None
        Specify validate data set size.
        float value specifies fraction of input data set, int value specifies exact number of data instances

    stratified : boolean, default: False
        Define whether sampling should be stratified, according to label value.

    shuffle : boolean, default : True
        Define whether do shuffle before splitting or not.

    split_points : None, list, default : None
        Specify the point(s) by which continuous label values are bucketed into bins for stratified split.
        eg.[0.2] for two bins or [0.1, 1, 3] for 4 bins

    """

    def __init__(self, random_state=None, test_size=None, train_size=None, validate_size=None, stratified=False,
                 shuffle=True, split_points=True):
        super(DataSplitParam, self).__init__()
        self.random_state = random_state
        self.test_size = test_size
        self.train_size = train_size
        self.validate_size = validate_size
        self.stratified = stratified
        self.shuffle = shuffle
        self.split_points = split_points

    def check(self):
        model_param_descr = "cross validation param's "
        if self.random_state is not None:
            if isinstance(self.random_state, int):
                raise ValueError(f"{model_param_descr} random state should be int type")

        if self.test_size is not None:
            self.check_nonnegative_number(self.test_size, model_param_descr)
            if isinstance(self.test_size, float):
                self.check_decimal_float(self.test_size, model_param_descr)
        if self.train_size is not None:
            self.check_nonnegative_number(self.train_size, model_param_descr)
            if isinstance(self.train_size, float):
                self.check_decimal_float(self.train_size, model_param_descr)
        if self.validate_size is not None:
            self.check_nonnegative_number(self.validate_size, model_param_descr)
            if isinstance(self.validate_size, float):
                self.check_decimal_float(self.validate_size, model_param_descr)

        self.check_boolean(self.stratified, model_param_descr)
        self.check_boolean(self.shuffle, model_param_descr)

        if self.split_points is not None:
            if isinstance(self.split_points, list):
                raise ValueError(f"{model_param_descr} split_points should be list type")

        LOGGER.debug("Finish data_split parameter check!")
        return True
