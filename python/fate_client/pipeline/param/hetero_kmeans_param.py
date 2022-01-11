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

from pipeline.param.base_param import BaseParam


class KmeansParam(BaseParam):
    """
    Parameters used for K-means.
    ----------
    k : int, should be larger than 1 ,default 5.
        The number of the centroids to generate.
    max_iter : int, default 300.
        Maximum number of iterations of the hetero-k-means algorithm to run.
    tol : float, default 0.001。

    """

    def __init__(self, k=5, max_iter=300, tol=0.001):
        super(KmeansParam, self).__init__()
        self.k = k
        self.max_iter = max_iter
        self.tol = tol

    def check(self):
        descr = "Kmeans_param's"

        if not isinstance(self.k, int):
            raise ValueError(
                descr + "k {} not supported, should be int type".format(self.k))
        elif self.k <= 1:
            raise ValueError(
                descr + "k {} not supported, should be larger than 1")

        if not isinstance(self.max_iter, int):
            raise ValueError(
                descr + "max_iter not supported, should be int type".format(self.max_iter))
        elif self.max_iter <= 0:
            raise ValueError(
                descr + "max_iter not supported, should be larger than 0".format(self.max_iter))

        if not isinstance(self.tol, (float, int)):
            raise ValueError(
                descr + "tol not supported, should be float type".format(self.tol))
        elif self.tol < 0:
            raise ValueError(
                descr + "tol not supported, should be larger than or equal to 0".format(self.tol))
