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

from federatedml.feature.feature_selection.filter_base import BaseFilterMethod
from federatedml.feature.feature_selection.model_adapter.isometric_model import IsometricModel
from federatedml.framework.hetero.sync import selection_info_sync
from federatedml.param.feature_selection_param import CommonFilterParam
from federatedml.protobuf.generated import feature_selection_meta_pb2
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.util import fate_operator
from federatedml.util import anonymous_generator
from federatedml.util.component_properties import ComponentProperties


class IsoModelFilter(BaseFilterMethod):

    def __init__(self, filter_param, iso_model: IsometricModel):
        self.iso_model = iso_model
        super(IsoModelFilter, self).__init__(filter_param)

    def _parse_filter_param(self, filter_param: CommonFilterParam):
        self.metrics = filter_param.metrics[0]
        LOGGER.debug(f"In parse filter param, metrics: {filter_param.metrics}")
        if self.metrics not in self.iso_model.valid_value_name:
            raise ValueError(f"Metric {self.metrics} is not in this model's valid_value_name")
        self.filter_type = filter_param.filter_type[0]
        self.take_high = filter_param.take_high[0]
        self.threshold = filter_param.threshold[0]
        self.select_federated = filter_param.select_federated[0]
        self._validation_check()

    def get_meta_obj(self):
        LOGGER.debug(f"metrics: {self.metrics}, filter_type: {self.filter_type},"
                     f"take_high: {self.take_high}, threshold: {self.threshold},"
                     f"select_federated: {self.select_federated}")
        result = feature_selection_meta_pb2.FilterMeta(
            metrics=self.metrics,
            filter_type=self.filter_type,
            take_high=self.take_high,
            threshold=self.threshold,
            select_federated=self.select_federated
        )
        return result

    def _validation_check(self):
        if self.metrics == consts.IV:
            if not self.take_high:
                raise ValueError("Iv filter should take higher iv columns")

    def fit(self, data_instances, suffix):
        m = self.metrics
        metric_info = self.iso_model.get_metric_info(m)
        all_feature_values = metric_info.get_partial_values(
            self.selection_properties.select_col_names)
        col_names = [x for x in self.selection_properties.select_col_names]

        # all_feature_values = np.array(metric_info.values)
        # col_names = [x for x in metric_info.col_names]

        filter_type = self.filter_type
        take_high = self.take_high
        threshold = self.threshold
        if filter_type == "threshold":
            results = self._threshold_fit(all_feature_values, threshold, take_high)
        elif filter_type == "top_k":
            results = self._top_k_fit(all_feature_values, threshold, take_high)
        else:
            results = self._percentile_fit(all_feature_values, threshold, take_high)

        for v_idx, v in enumerate(all_feature_values):
            col_name = col_names[v_idx]
            self.selection_properties.add_feature_value(col_name, v)
            if v_idx in results:
                self.selection_properties.add_left_col_name(col_name)
        self._keep_one_feature(pick_high=take_high)

        return self

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
        self.role = role
        self.cpp = cpp
        super(FederatedIsoModelFilter, self).__init__(filter_param, iso_model)
        self.sync_obj = None

    @property
    def party_id(self):
        return self.cpp.local_partyid

    def _parse_filter_param(self, filter_param: CommonFilterParam):
        super()._parse_filter_param(filter_param)
        if filter_param.host_thresholds is None:
            self.host_threshold = None
        else:
            self.host_threshold = filter_param.host_thresholds[0]
        if isinstance(self.host_threshold, float):
            self.host_threshold = [self.host_threshold] * len(self.cpp.host_party_idlist)

        if self.role == consts.GUEST:
            self.host_selection_properties = []

    def fit(self, data_instances, suffix):
        self._sync_select_info(suffix)
        if self.role == consts.GUEST:
            self._guest_fit(suffix)
        else:
            self._host_fit(suffix)
        self._keep_one_feature(pick_high=self.take_high)
        return self

    def _guest_fit(self, suffix):
        m = self.metrics

        # for idx, m in enumerate(self.metrics):
        value_obj = self.iso_model.get_metric_info(m)
        self._fix_with_value_obj(value_obj, suffix)

    def _fix_with_value_obj(self, value_obj, suffix):
        all_feature_values = value_obj.get_partial_values(self.selection_properties.select_col_names)
        col_names = [("guest", x) for x in self.selection_properties.select_col_names]
        if self.select_federated:
            # all_feature_values, col_names = value_obj.union_result()
            host_threshold = {}
            for idx, host_party_id in enumerate(value_obj.host_party_ids):
                # host_id = self.cpp.host_party_idlist.index(int(host_party_id))
                host_property = self.host_selection_properties[idx]
                all_feature_values.extend(value_obj.get_partial_values(
                    host_property.select_col_names, host_party_id
                ))
                col_names.extend([(host_party_id, x) for x in host_property.select_col_names])
                if self.host_threshold is None:
                    host_threshold[host_party_id] = self.threshold
                else:
                    host_threshold[host_party_id] = self.host_threshold[idx]
        else:
            # all_feature_values = value_obj.get_values()
            # col_names = value_obj.get_col_names()
            host_threshold = None

        filter_type = self.filter_type
        take_high = self.take_high
        threshold = self.threshold

        if filter_type == "threshold":
            results = self._threshold_fit(all_feature_values, threshold,
                                          take_high, host_threshold, col_names)
        elif filter_type == "top_k":
            results = self._top_k_fit(all_feature_values, threshold, take_high)
        else:
            results = self._percentile_fit(all_feature_values, threshold, take_high)
        # LOGGER.debug(f"results length: {len(results)}, type of results is: {type(results)}")
        results = set(results)
        # LOGGER.debug(f"filter_type: {filter_type}, results: {results}, "
        #              f"all_feature_values: {all_feature_values}")

        for v_idx, v in enumerate(all_feature_values):
            # LOGGER.debug(f"all_feature_values: {all_feature_values},"
            # f"col_names: {col_names},"
            #             f"v_idx: {v_idx}")
            col_name = col_names[v_idx]
            if col_name[0] == consts.GUEST:
                self.selection_properties.add_feature_value(col_name[1], v)
                if v_idx in results:
                    self.selection_properties.add_left_col_name(col_name[1])
            else:
                # LOGGER.debug(f"host_selection_propertied: {self.host_selection_properties}")
                # LOGGER.debug(f" col_name: {col_name}")
                host_idx = self.cpp.host_party_idlist.index(int(col_name[0]))
                # LOGGER.debug(f"header: {self.host_selection_properties[host_idx].header}")
                host_prop = self.host_selection_properties[host_idx]
                # if len(self.metrics) == 1:
                host_prop.add_feature_value(col_name[1], v)

                if v_idx in results:
                    host_prop.add_left_col_name(col_name[1])
        for host_prop in self.host_selection_properties:
            self._keep_one_feature(pick_high=self.take_high, selection_properties=host_prop,
                                   feature_values=host_prop.feature_values)

        if self.select_federated:
            self.sync_obj.sync_select_results(self.host_selection_properties, suffix=suffix)

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

    def get_meta_obj(self):
        result = feature_selection_meta_pb2.FilterMeta(
            metrics=self.metrics,
            filter_type=self.filter_type,
            take_high=self.take_high,
            threshold=self.threshold,
            select_federated=self.select_federated
        )
        return result

    def _host_fit(self, suffix):
        if not self.select_federated:
            for col_name in self.selection_properties.last_left_col_names:
                self.selection_properties.add_left_col_name(col_name)
            return
        self.sync_obj.sync_select_results(self.selection_properties,
                                          decode_func=self.decode_func,
                                          suffix=suffix)
        # LOGGER.debug("In fit selected result, left_col_names: {}".format(self.selection_properties.left_col_names))
        return self

    def decode_func(self, encoded_name):
        fid = anonymous_generator.reconstruct_fid(encoded_name)
        return self.selection_properties.header[fid]

    def _sync_select_info(self, suffix):
        if not self.select_federated:
            return
        if self.role == consts.GUEST:
            assert isinstance(self.sync_obj, selection_info_sync.Guest)
            self.host_selection_properties = self.sync_obj.sync_select_cols(suffix=suffix)
        else:
            encoded_names = []
            for col_name in self.selection_properties.select_col_names:
                fid = self.selection_properties.col_name_maps[col_name]
                encoded_names.append(anonymous_generator.generate_anonymous(
                    fid=fid, role=self.role, party_id=self.party_id
                ))
            # LOGGER.debug(f"Before send, encoded_names: {encoded_names},"
            #             f"select_names: {self.selection_properties.select_col_names}")
            self.sync_obj.sync_select_cols(encoded_names, suffix=suffix)

    def set_transfer_variable(self, transfer_variable):
        if self.role == consts.GUEST:
            self.sync_obj = selection_info_sync.Guest()
        else:
            self.sync_obj = selection_info_sync.Host()
        self.sync_obj.register_selection_trans_vars(transfer_variable)
