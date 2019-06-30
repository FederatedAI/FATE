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
from federatedml.util import consts


class FeatureBinningParam(BaseParam):
    """
    Define the feature binning method

    Parameters
    ----------
    process_method : str, 'fit' or 'transform', default: "fit"
        Specify what process to do.

    method : str, 'quantile', default: 'quantile'
        Binning method.

    compress_thres: int, default: 10000
        When the number of saved summaries exceed this threshold, it will call its compress function

    head_size: int, default: 10000
        The buffer size to store inserted observations. When head list reach this buffer size, the
        QuantileSummaries object start to generate summary(or stats) and insert into its sampled list.

    error: float, 0 <= error < 1 default: 0.001
        The error of tolerance of binning. The final split point comes from original data, and the rank
        of this value is close to the exact rank. More precisely,
        floor((p - 2 * error) * N) <= rank(x) <= ceil((p + 2 * error) * N)
        where p is the quantile in float, and N is total number of data.

    bin_num: int, bin_num > 0, default: 10
        The max bin number for binning

    cols : list of string or int, default: -1
        Specify which columns need to calculated. -1 represent for all columns. If you need to indicate specific
        cols, provide a list of header string instead of -1.

    adjustment_factor : float, default: 0.5
        the adjustment factor when calculating WOE. This is useful when there is no event or non-event in
        a bin.

    local_only : bool, default: False
        Whether just provide binning method to guest party. If true, host party will do nothing.

    display_result : list, default: ['iv']
        Specify what results to show. The available results include:
        ['iv', 'woe_array', 'iv_array', 'event_count_array', 'non_event_count_array', 'event_rate_array',
        'non_event_rate_array', 'is_woe_monotonic', 'bin_nums', 'split_points']
        for each features

    """

    def __init__(self, process_method='fit',
                 method=consts.QUANTILE, compress_thres=consts.DEFAULT_COMPRESS_THRESHOLD,
                 head_size=consts.DEFAULT_HEAD_SIZE,
                 error=consts.DEFAULT_RELATIVE_ERROR,
                 bin_num=consts.G_BIN_NUM, cols=-1, adjustment_factor=0.5,
                 local_only=False,
                 # meta_table='binning_meta_table',
                 # param_table='binning_param_table',
                 # transform_table='binning_transform_table',
                 # result_namespace='binning_namespace',
                 display_result='simple'):
        self.process_method = process_method
        self.method = method
        self.compress_thres = compress_thres
        self.head_size = head_size
        self.error = error
        self.adjustment_factor = adjustment_factor
        self.bin_num = bin_num
        self.cols = cols
        self.local_only = local_only

        if display_result == 'simple':
            display_result = ['iv']
        self.display_result = display_result

    def check(self):
        descr = "hetero binning param's"
        self.check_string(self.method, descr)
        self.method = self.method.lower()
        self.check_valid_value(self.method, descr, [consts.QUANTILE])
        self.check_positive_integer(self.compress_thres, descr)
        self.check_positive_integer(self.head_size, descr)
        self.check_decimal_float(self.error, descr)
        self.check_positive_integer(self.bin_num, descr)
        self.check_defined_type(self.cols, descr, ['list', 'int', 'RepeatedScalarContainer'])
        self.check_open_unit_interval(self.adjustment_factor, descr)
        self.check_defined_type(self.display_result, descr, ['list'])
        for idx, d_s in enumerate(self.display_result):
            self.display_result[idx] = self.check_and_change_lower(d_s,
                                                                   ['iv', 'woe_array', 'iv_array',
                                                                    'event_count_array', 'non_event_count_array',
                                                                    'event_rate_array', 'bin_nums', 'split_points',
                                                                    'non_event_rate_array', 'is_woe_monotonic'],
                                                                   descr)
