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

from federatedml.param.base_param import BaseParam


class InitParam(BaseParam):
    """
    Initialize Parameters used in initializing a model.

    Parameters
    ----------
    init_method : str, 'random_uniform', 'random_normal', 'ones', 'zeros' or 'const'. default: 'random_uniform'
        Initial method.

    init_const : int or float, default: 1
        Required when init_method is 'const'. Specify the constant.

    fit_intercept : bool, default: True
        Whether to initialize the intercept or not.

    """

    def __init__(self, init_method='random_uniform', init_const=1, fit_intercept=True, random_seed=None):
        super().__init__()
        self.init_method = init_method
        self.init_const = init_const
        self.fit_intercept = fit_intercept
        self.random_seed = random_seed

    def check(self):
        if type(self.init_method).__name__ != "str":
            raise ValueError(
                "Init param's init_method {} not supported, should be str type".format(self.init_method))
        else:
            self.init_method = self.init_method.lower()
            if self.init_method not in ['random_uniform', 'random_normal', 'ones', 'zeros', 'const']:
                raise ValueError(
                    "Init param's init_method {} not supported, init_method should in 'random_uniform',"
                    " 'random_normal' 'ones', 'zeros' or 'const'".format(self.init_method))

        if type(self.init_const).__name__ not in ['int', 'float']:
            raise ValueError(
                "Init param's init_const {} not supported, should be int or float type".format(self.init_const))

        if type(self.fit_intercept).__name__ != 'bool':
            raise ValueError(
                "Init param's fit_intercept {} not supported, should be bool type".format(self.fit_intercept))

        if self.random_seed is not None:
            if type(self.random_seed).__name__ != 'int':
                raise ValueError(
                    "Init param's random_seed {} not supported, should be int or float type".format(self.random_seed))

        return True