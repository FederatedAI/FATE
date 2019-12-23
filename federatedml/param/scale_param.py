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
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class ScaleParam(BaseParam):
    """
    Define the feature scale parameters.

    Parameters
    ----------
        method : str, like scale in sklearn, now it support "min_max_scale" and "standard_scale", and will support other scale method soon.
                 Default None, which will do nothing for scale

        mode: str, the mode support "normal" and "cap". for mode is "normal", the feat_upper and feat_lower is the normal value like "10" or "3.1" and for "cap", feat_upper and
              feature_lower will between 0 and 1, which means the percentile of the column. Default "normal"

        area: str, It supports "all" and "col". For "all", it will scale all data column, and for "col",
        it just scale ths columns which parameter "cale_column_idx" corresponding to, so "scale_column_idx" will be a list, which including the column idx to be scaled.
        Default "all"

        feat_upper: int or float, the upper limit in the column. If the scaled value is larger than feat_upper, it will be set to feat_upper. Default None.
        feat_lower: int or float, the lower limit in the column. If the scaled value is less than feat_lower, it will be set to feat_lower. Default None.

        scale_column_idx: list, while parameter "area" is "col", the idx of column in scale_column_idx will be scaled, while the idx of column is not in, it will not be scaled.

        with_mean: bool, used for "standard_scale". Default False.
        with_std: bool, used for "standard_scale". Default False.
            The standard scale of column x is calculated as : z = (x - u) / s, where u is the mean of the column and s is the standard deviation of the column.
            if with_mean is False, u will be 0, and if with_std is False, s will be 1.

        need_run: bool, default True
            Indicate if this module needed to be run

    """

    def __init__(self, method=None, mode="normal", area="all", scale_column_idx=None, feat_upper=None, feat_lower=None,
                 with_mean=True, with_std=True, need_run=True):
        super().__init__()
        self.method = method
        self.mode = mode
        self.area = area
        self.feat_upper = feat_upper
        self.feat_lower = feat_lower
        self.scale_column_idx = scale_column_idx

        self.with_mean = with_mean
        self.with_std = with_std

        self.need_run = need_run

    def check(self):
        if self.method is not None:
            descr = "scale param's method"
            self.method = self.check_and_change_lower(self.method,
                                                      [consts.MINMAXSCALE, consts.STANDARDSCALE],
                                                      descr)

        descr = "scale param's mode"
        self.mode = self.check_and_change_lower(self.mode,
                                                [consts.NORMAL, consts.CAP],
                                                descr)

        descr = "scale param's area"
        self.area = self.check_and_change_lower(self.area,
                                                [consts.ALL, consts.COL],
                                                descr)

        self.check_boolean(self.with_mean, "scale_param with_mean")
        self.check_boolean(self.with_std, "scale_param with_std")

        LOGGER.debug("Finish scale parameter check!")
        return True
