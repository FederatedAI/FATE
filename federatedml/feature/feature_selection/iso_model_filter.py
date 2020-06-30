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

import math

import numpy as np

from arch.api.utils import log_utils
from federatedml.feature.feature_selection.filter_base import BaseFilterMethod
from federatedml.feature.hetero_feature_selection.isometric_model import IsometricModel
from federatedml.framework.hetero.sync import selection_info_sync
from federatedml.param.feature_selection_param import CommonFilterParam
from federatedml.util import consts
from federatedml.util import fate_operator
from federatedml.util.component_properties import ComponentProperties

LOGGER = log_utils.getLogger()


class IsoModelFilter(BaseFilterMethod):

    def __init__(self, filter_param, iso_model: IsometricModel):
        self.iso_model = iso_model
        super(IsoModelFilter, self).__init__(filter_param)

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

    def get_meta_obj(self, meta_dicts):
        pass

    def _validation_check(self):
        return True

    def fit(self, data_instances, suffix):
        for idx, m in enumerate(self.metrics):
            value_obj = self.iso_model.feature_values[m]
            all_feature_values = value_obj.values
            col_names = [x for x in value_obj.col_names]

            filter_type = self.filter_type[idx]
            take_high = self.take_high[idx]
            threshold = self.threshold[idx]
            if filter_type == "threshold":
                results = self._threshold_fit(all_feature_values, threshold, take_high)
            elif filter_type == "top_k":
                results = self._top_k_fit(all_feature_values, threshold, take_high)
            else:
                results = self._percentile_fit(all_feature_values, threshold, take_high)

            for v_idx, v in enumerate(all_feature_values):
                col_name = col_names[v_idx]
                self.selection_properties.add_feature_value(col_name, v)
                if idx in results:
                    self.selection_properties.add_left_col_name(col_name)

    def _threshold_fit(self, values, threshold, take_high):
        result = []
        for idx, v in enumerate(values):
            if take_high:
                if v >= threshold:
                    result.append(idx)
            else:
                if v <= threshold:
                    result.append(idx)
        return result

    def _top_k_fit(self, values, k, take_high):
        sorted_idx = np.argsort(values)
        result = []
        if take_high:
            for idx in sorted_idx[::-1]:
                result.append(idx)
                if len(result) >= k:
                    break
        else:
            for idx in sorted_idx:
                result.append(idx)
                if len(result) >= k:
                    break
        return result

    def _percentile_fit(self, values, percent, take_high):
        k = math.ceil(percent * len(values))
        return self._top_k_fit(values, k, take_high)


class FederatedIsoModelFilter(IsoModelFilter):

    def __init__(self, filter_param, iso_model: IsometricModel, role, cpp: ComponentProperties):
        super(FederatedIsoModelFilter, self).__init__(filter_param, iso_model)
        self.role = role
        self.cpp = cpp
        self.sync_obj = None

    @property
    def party_id(self):
        return self.cpp.local_partyid

    def _parse_filter_param(self, filter_param: CommonFilterParam):
        super()._parse_filter_param(filter_param)
        self.host_threshold = filter_param.host_thresholds

    def fit(self, data_instances, suffix):
        pass

    def _guest_fit(self):
        for idx, m in enumerate(self.metrics):
            value_obj = self.iso_model.feature_values[m]
            all_feature_values = value_obj.values
            col_names = [("guest", x) for x in value_obj.col_names]

            for host_party_id, host_values in self.iso_model.host_values.items():
                host_id = self.cpp.host_party_idlist.index(host_party_id)
                value_obj = host_values[m]
                all_feature_values.extend(value_obj.values)
                col_names.extend([(host_id, x) for x in value_obj.col_names])

            filter_type = self.filter_type[idx]
            take_high = self.take_high[idx]
            threshold = self.threshold[idx]
            host_threshold = self.host_threshold[idx]
            if filter_type == "threshold":
                results = self._threshold_fit(all_feature_values, threshold,
                                              take_high, host_threshold, col_names)
            elif filter_type == "top_k":
                results = self._top_k_fit(all_feature_values, threshold, take_high)
            else:
                results = self._percentile_fit(all_feature_values, threshold, take_high)

            for v_idx, v in enumerate(all_feature_values):
                col_name = col_names[v_idx]
                if col_name[0] == consts.GUEST:
                    self.selection_properties.add_feature_value(col_name[1], v)
                    if idx in results:
                        self.selection_properties.add_left_col_name(col_name[1])
                else:
                    host_prop = self.host_selection_properties[col_name[0]]
                    host_prop.add_feature_value(col_name[1], v)
                    if idx in results:
                        host_prop.add_left_col_name(col_name[1])

    def _threshold_fit(self, values, threshold, take_high,
                       host_thresholds=None, col_names=None):
        if host_thresholds is None:
            return super()._threshold_fit(values, threshold, take_high)
        result = []
        for idx, v in enumerate(values):
            party = col_names[idx][0]
            if party == 'guest':
                thres = threshold
            else:
                thres = host_thresholds[party]

            if take_high:
                if v >= thres:
                    result.append(idx)
            else:
                if v <= thres:
                    result.append(idx)
        return result


    def get_meta_obj(self, meta_dicts):
        pass

    def _host_fit(self, suffix):
        if not self.select_federated:
            for col_name in self.selection_properties.last_left_col_names:
                self.selection_properties.add_left_col_name(col_name)
            return
        self.sync_obj.sync_select_results(self.selection_properties,
                                          decode_func=self.decode_func,
                                          suffix=suffix)
        LOGGER.debug("In fit selected result, left_col_names: {}".format(self.selection_properties.left_col_names))
        return self

    def decode_func(self, encoded_name):
        fid = fate_operator.reconstruct_fid(encoded_name)
        return self.selection_properties.header[fid]

    def _sync_select_info(self, suffix):
        if not self.select_federated:
            return
        if self.role == consts.GUEST:
            assert isinstance(self.sync_obj, selection_info_sync.Guest)
            self.host_selection_properties = self.sync_obj.sync_select_cols(suffix=suffix)
        else:
            encoded_names = []
            for fid, col_name in enumerate(self.selection_properties.select_col_names):
                encoded_names.append(fate_operator.generate_anonymous(
                    fid=fid, role=self.role, party_id=self.party_id
                ))
            self.sync_obj.sync_select_cols(encoded_names, suffix=suffix)

    def set_transfer_variable(self, transfer_variable):
        if self.role == consts.GUEST:
            self.sync_obj = selection_info_sync.Guest()
        else:
            self.sync_obj = selection_info_sync.Host()
        self.sync_obj.register_selection_trans_vars(transfer_variable)
