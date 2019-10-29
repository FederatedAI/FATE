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
        method : str, now it support "min_max_scale" and "standard_scale", and will support other scale method soon.
                 Default None, which will do nothing for scale

        mode: str, for method is "min_max_scale" and for "standard_scale" it is useless, the mode just support "normal" now, and will support "cap" mode in the furture.
              for mode is "min_max_scale", the feat_upper and feat_lower is the normal value and for "cap", feat_upper and
              feature_lower will between 0 and 1, which means the percentile of the column. Default "normal"

        area: str, for method is "min_max_scale" and for "standard_scale" it is useless. It supports "all" and "col". For "all",
            feat_upper/feat_lower will act on all data column, so it will just be a value, and for "col", it just acts
            on one column they corresponding to, so feat_lower/feat_upper will be a list, which size will equal to the number of columns

        feat_upper: int or float, used for "min_max_scale", the upper limit in the column. If the scaled value is larger than feat_upper, it will be set to feat_upper. Default None.
        feat_lower: int or float, used for "min_max_scale", the lower limit in the column. If the scaled value is less than feat_lower, it will be set to feat_lower. Default None.

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
        # if self.area == consts.ALL:
        #     if self.feat_lower is not None:
        #         if type(self.feat_lower).__name__ not in ["float", "int"]:
        #             raise ValueError(
        #                 "scale param's feat_lower {} not supported, should be float or int type".format(
        #                     self.feat_lower))
        #
        #     if self.feat_upper is not None:
        #         if type(self.feat_upper).__name__ not in ["float", "int"]:
        #             raise ValueError(
        #                 "scale param's feat_upper {} not supported, should be float or int type".format(
        #                     self.feat_upper))
        #
        # elif self.area == consts.COL:
        #     descr = "scale param's feat_lower"
        #     self.check_defined_type(self.feat_lower, descr, ['list'])
        #
        #     descr = "scale param's feat_upper"
        #     self.check_defined_type(self.feat_upper, descr, ['list'])


        self.check_boolean(self.with_mean, "scale_param with_mean")
        self.check_boolean(self.with_std, "scale_param with_std")

        LOGGER.debug("Finish scale parameter check!")
        return True
