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


import re

from federatedml.param.base_param import BaseParam
from federatedml.util import consts


class StatisticsParam(BaseParam):
    """
    Define statistics params

    Parameters
    ----------
    statistics: list, string, default "summary"
        Specify the statistic types to be computed.
        "summary" represents list: [consts.SUM, consts.MEAN, consts.STANDARD_DEVIATION,
                    consts.MEDIAN, consts.MIN, consts.MAX,
                    consts.MISSING_COUNT, consts.SKEWNESS, consts.KURTOSIS]
        "describe" represents list: [consts.COUNT, consts.MEAN,
                    consts.STANDARD_DEVIATION, consts.MIN, consts.MAX]

    column_names: list of string, default []
        Specify columns to be used for statistic computation by column names in header

    column_indexes: list of int, default -1
        Specify columns to be used for statistic computation by column order in header
        -1 indicates to compute statistics over all columns

    bias: bool, default: True
        If False, the calculations of skewness and kurtosis are corrected for statistical bias.

    need_run: bool, default True
        Indicate whether to run this modules
    """

    LEGAL_STAT = [consts.COUNT, consts.SUM, consts.MEAN, consts.STANDARD_DEVIATION,
                  consts.MEDIAN, consts.MIN, consts.MAX, consts.VARIANCE,
                  consts.COEFFICIENT_OF_VARIATION, consts.MISSING_COUNT,
                  consts.SKEWNESS, consts.KURTOSIS]
    LEGAL_QUANTILE = re.compile("^(100)|([1-9]?[0-9])%$")

    def __init__(self, statistics="summary", column_names=None,
                 column_indexes=-1, need_run=True, abnormal_list=None,
                 quantile_error=consts.DEFAULT_RELATIVE_ERROR, bias=True):
        super().__init__()
        self.statistics = statistics
        self.column_names = column_names
        self.column_indexes = column_indexes
        self.abnormal_list = abnormal_list
        self.need_run = need_run
        self.quantile_error = quantile_error
        self.bias = bias
        if column_names is None:
            self.column_names = []
        if column_indexes is None:
            self.column_indexes = []
        if abnormal_list is None:
            self.abnormal_list = []

    @staticmethod
    def extend_statistics(statistic_name):
        if statistic_name == "summary":
            return [consts.SUM, consts.MEAN, consts.STANDARD_DEVIATION,
                    consts.MEDIAN, consts.MIN, consts.MAX,
                    consts.MISSING_COUNT, consts.SKEWNESS, consts.KURTOSIS,
                    consts.COEFFICIENT_OF_VARIATION]
        if statistic_name == "describe":
            return [consts.COUNT, consts.MEAN, consts.STANDARD_DEVIATION, consts.MIN, consts.MAX]

    @staticmethod
    def find_stat_name_match(stat_name):
        if stat_name in StatisticsParam.LEGAL_STAT or StatisticsParam.LEGAL_QUANTILE.match(stat_name):
            return True
        return False

        # match_result = [legal_name == stat_name for legal_name in StatisticsParam.LEGAL_STAT]
        # match_result.append(0 if LEGAL_QUANTILE.match(stat_name) is None else True)
        # match_found = sum(match_result) > 0
        # return match_found

    def check(self):
        model_param_descr = "Statistics's param statistics"
        BaseParam.check_boolean(self.need_run, model_param_descr)
        if not isinstance(self.statistics, list):
            if self.statistics in [consts.DESCRIBE, consts.SUMMARY]:
                self.statistics = StatisticsParam.extend_statistics(self.statistics)
            else:
                self.statistics = [self.statistics]

        for stat_name in self.statistics:
            match_found = StatisticsParam.find_stat_name_match(stat_name)
            if not match_found:
                raise ValueError(f"Illegal statistics name provided: {stat_name}.")

        model_param_descr = "Statistics's param column_names"
        if not isinstance(self.column_names, list):
            raise ValueError(f"column_names should be list of string.")
        for col_name in self.column_names:
            BaseParam.check_string(col_name, model_param_descr)

        model_param_descr = "Statistics's param column_indexes"
        if not isinstance(self.column_indexes, list) and self.column_indexes != -1:
            raise ValueError(f"column_indexes should be list of int or -1.")
        if self.column_indexes != -1:
            for col_index in self.column_indexes:
                if not isinstance(col_index, int):
                    raise ValueError(f"{model_param_descr} should be int or list of int")
                if col_index < -consts.FLOAT_ZERO:
                    raise ValueError(f"{model_param_descr} should be non-negative int value(s)")

        if not isinstance(self.abnormal_list, list):
            raise ValueError(f"abnormal_list should be list of int or string.")

        self.check_decimal_float(self.quantile_error, "Statistics's param quantile_error ")
        self.check_boolean(self.bias, "Statistics's param bias ")
        return True
