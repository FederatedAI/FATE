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

from federatedml.feature.binning.base_binning import Binning
from federatedml.statistic.statics import MultivariateStatisticalSummary

class BucketBinning(Binning):
    """
    For bucket binning, the length of each bin is the same which is:
    L = [max(x) - min(x)] / n

    The split points are min(x) + L * k
    where k is the index of a bin.
    """

    def __init__(self, params):
        super(BucketBinning, self).__init__(params)

    def fit_split_points(self, data_instances):
        """
        Apply the binning method

        Parameters
        ----------
        data_instances : DTable
            The input data

        cols : int or list of int
            Specify which column(s) need to apply binning. -1 means do binning for all columns.

        Returns
        -------
        split_points, 2-dimension list.
            Each row represent for the split points for a feature. The element in each row represent for
            the corresponding split point.
            e.g.
            split_points = [[0.1, 0.2, 0.3, 0.4 ...],    # The first feature
                            [1, 2, 3, 4, ...],           # The second feature
                            ...]                         # Other features

        """
        self._init_cols(data_instances)

        statistics = MultivariateStatisticalSummary(data_instances, cols)
        split_points = []
        max_list = statistics.get_max(cols)
        min_list = statistics.get_min(cols)
        n = data_instances.count()
        for idx, max_value in enumerate(max_list):
            min_value = min_list[idx]
            split_point = []
            L = (max_value - min_value) / n
            for k in range(self.bin_num - 1):
                s_p = min_value + (k + 1) * L
                split_point.append(s_p)
            split_points.append(split_point)
        return split_points