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
#  limitations under the License
#

from federatedml.statistic.data_overview import get_header


class BaseCorrelation(object):

    def __init__(self, params):
        self.cols = params.cols
        self.cols_dict = {}

    def compute_correlation(self, data_instances):
        """
        Compute correlation between data_instances

        Parameters
        ----------
        data_instances : DTable
            The input data
        """
        pass

    def _init_cols(self, data_instances):
        header = get_header(data_instances)
        if self.cols == -1:
            self.cols = header

        self.cols_dict = {}
        for col in self.cols:
            col_index = header.index(col)
            self.cols_dict[col] = col_index