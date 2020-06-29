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

from federatedml.feature.feature_selection.filter_base import BaseFilterMethod
from federatedml.feature.hetero_feature_selection.isometric_model import IsometricModel
from federatedml.framework.hetero.sync import selection_info_sync
from federatedml.param.feature_selection_param import CommonFilterParam
from federatedml.util import consts


class IsoModelFilter(BaseFilterMethod):

    def __init__(self, filter_param, role, iso_model: IsometricModel):
        super(IsoModelFilter, self).__init__(filter_param)
        self.role = role
        self.iso_model = iso_model
        if self.role == consts.GUEST:
            self.sync_obj = selection_info_sync.Guest()
        elif self.role == consts.HOST:
            self.sync_obj = selection_info_sync.Host()
        else:
            raise ValueError(f"Feature selection do not need role: {self.role}")
        self._suffix_list = None
        self.host_selection_properties = None

    def _parse_filter_param(self, filter_param: CommonFilterParam):
        self.metrics = filter_param.metrics
        for m in self.metrics:
            if m not in self.iso_model.valid_value_name:
                raise ValueError(f"Metric {m} is not in this model's valid_value_name")
        self.filter_type = filter_param.filter_type
        self.take_high = filter_param.take_high
        self.threshold = filter_param.threshold
        self.select_federated = filter_param.select_federated
        self._validation_check()

    def apply_suffix(self, suffix):
        self._suffix_list = suffix

    def fit(self, data_instances, suffix):
        self.host_selection_properties = self.sync_obj.sync_select_cols(suffix=suffix)


    def get_meta_obj(self, meta_dicts):
        pass

    def _validation_check(self):
        return True
