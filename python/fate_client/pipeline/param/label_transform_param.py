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
import copy

from pipeline.param.base_param import BaseParam
from federatedml.param.positive_unlabeled_param import PositiveUnlabeledParam


class LabelTransformParam(BaseParam):
    """
    Define label transform param that used in label transform.

    Parameters
    ----------

    label_encoder : None or dict, default : None
        Specify (label, encoded label) key-value pairs for transforming labels to new values.
        e.g. {"Yes": 1, "No": 0}

    label_list : None or list, default : None
        List all input labels, used for matching types of original keys in label_encoder dict,
        length should match key count in label_encoder
        e.g. ["Yes", "No"]

    unlabeled_digit: None or integer, default: None
        Whether it has unlabeled data. If true, declaring the unlabeled digit.

    need_run: bool, default: True
        Specify whether to run label transform

    pu_param: PositiveUnlabeledParam object, default: default PositiveUnlabeledParam object
        positive unlabeled param

    """

    def __init__(self, label_encoder=None, label_list=None, unlabeled_digit=None, need_run=True,
                 pu_param=PositiveUnlabeledParam()):
        super(LabelTransformParam, self).__init__()
        self.label_encoder = label_encoder
        self.label_list = label_list
        self.unlabeled_digit = unlabeled_digit
        self.need_run = need_run
        self.pu_param = copy.deepcopy(pu_param)

    def check(self):
        model_param_descr = "label transform param's "

        BaseParam.check_boolean(self.need_run, f"{model_param_descr} need run ")

        if self.label_encoder is not None:
            if not isinstance(self.label_encoder, dict):
                raise ValueError(f"{model_param_descr} label_encoder should be dict type")

        if self.label_list is not None:
            if not isinstance(self.label_list, list):
                raise ValueError(f"{model_param_descr} label_list should be list type")
            if self.label_encoder and len(self.label_list) != len(self.label_encoder.keys()):
                raise ValueError(f"label_list length should match label_encoder key count")

        if self.unlabeled_digit is not None and type(self.unlabeled_digit).__name__ != "int":
            raise ValueError(f"{model_param_descr} unlabeled_digit should be None or an integer")

        self.pu_param.check()

        return True
