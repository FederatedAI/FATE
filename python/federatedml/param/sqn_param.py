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


class StochasticQuasiNewtonParam(BaseParam):
    """
    Parameters used for stochastic quasi-newton method.

    Parameters
    ----------
    update_interval_L : int, default: 3
        Set how many iteration to update hess matrix

    memory_M : int, default: 5
        Stack size of curvature information, i.e. y_k and s_k in the paper.

    sample_size: int, default: 5000
        Sample size of data that used to update Hess matrix

    """
    def __init__(self, update_interval_L=3, memory_M=5, sample_size=5000, random_seed=None):
        super().__init__()
        self.update_interval_L = update_interval_L
        self.memory_M = memory_M
        self.sample_size = sample_size
        self.random_seed = random_seed

    def check(self):
        descr = "hetero sqn param's"
        self.check_positive_integer(self.update_interval_L, descr)
        self.check_positive_integer(self.memory_M, descr)
        self.check_positive_integer(self.sample_size, descr)
        if self.random_seed is not None:
            self.check_positive_integer(self.random_seed, descr)
        return True