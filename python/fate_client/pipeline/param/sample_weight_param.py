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
from pipeline.param import consts


class SampleWeightParam(BaseParam):
    """
    Define sample weight parameters.

    Parameters
    ----------

    class_weight : str or dict, default None
        class weight dictionary or class weight computation mode, string value only accepts 'balanced';
        If dict provided, key should be class(label), and weight will not be normalize, e.g.: {'0': 1, '1': 2}
        If both class_weight and sample_weight_name are None, return original input data

    sample_weight_name : str, name of column which specifies sample weight.
        feature name of sample weight; if both class_weight and sample_weight_name are None, return original input data

    normalize : bool, default False
        whether to normalize sample weight extracted from `sample_weight_name` column

    need_run : bool, default True
        whether to run this module or not

    """

    def __init__(self, class_weight=None, sample_weight_name=None, normalize=False, need_run=True):
        self.class_weight = class_weight
        self.sample_weight_name = sample_weight_name
        self.normalize = normalize
        self.need_run = need_run

    def check(self):

        descr = "sample weight param's"

        if self.class_weight:
            if not isinstance(self.class_weight, str) and not isinstance(self.class_weight, dict):
                raise ValueError(f"{descr} class_weight must be str, dict, or None.")
            if isinstance(self.class_weight, str):
                self.class_weight = self.check_and_change_lower(self.class_weight,
                                                                [consts.BALANCED],
                                                                f"{descr} class_weight")

        if self.sample_weight_name:
            self.check_string(self.sample_weight_name, f"{descr} sample_weight_name")

        self.check_boolean(self.need_run, f"{descr} need_run")
        self.check_boolean(self.normalize, f"{descr} normalize")

        return True
