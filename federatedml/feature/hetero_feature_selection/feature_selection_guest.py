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


from arch.api.utils import log_utils
from federatedml.feature import feature_selection
from federatedml.feature.hetero_feature_selection import filter_result
from federatedml.feature.hetero_feature_selection.base_feature_selection import BaseHeteroFeatureSelection
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroFeatureSelectionGuest(BaseHeteroFeatureSelection):
    def __init__(self):
        super(HeteroFeatureSelectionGuest, self).__init__()
        self.host_left_cols = {}
        self.static_obj = None
        self.flowid = ''
        self.party_name = consts.GUEST
        self.fed_filter_count = 0
        self.host_filter_result = filter_result.RemoteFilterResult()

    def fit(self, data_instances):
        LOGGER.info("Start Hetero Selection Fit and transform.")
        self._abnormal_detection(data_instances)
        self._init_cols(data_instances)

        for method in self.filter_methods:
            self.filter_one_method(data_instances, method)
            LOGGER.debug("After method: {}, left_cols: {}".format(method, self.filter_result.get_left_cols()))
            # self._add_left_cols()
            # self._renew_left_col_names()

        new_data = self._transfer_data(data_instances)
        LOGGER.info("Finish Hetero Selection Fit and transform.")
        return new_data

    def transform(self, data_instances):
        self._abnormal_detection(data_instances)
        self._transform_init_cols(data_instances)
        LOGGER.info("[Result][FeatureSelection][Guest]In transform, Self left cols are: {}".format(
            self.filter_result.get_left_cols()
        ))
        new_data = self._transfer_data(data_instances)
        return new_data

    def filter_one_method(self, data_instances, method):

        if method == consts.IV_VALUE_THRES:
            iv_param = self.model_param.iv_value_param

            if not self.local_only:
                host_select_cols = self._get_host_select_cols(consts.IV_VALUE_THRES)
                LOGGER.debug("In iv value filter, host_select_cols: {}".format(host_select_cols))
                iv_filter = feature_selection.IVValueSelectFilter(iv_param,
                                                                  self.filter_result.this_to_select_cols_index,
                                                                  self.binning_model,
                                                                  host_select_cols=host_select_cols)
                new_left_cols = iv_filter.fit(data_instances, fit_host=True)
                # self._renew_final_left_cols(new_left_cols)
                self.filter_result.add_left_col_index(new_left_cols)

                host_left_cols = iv_filter.host_cols
                left_cols = host_left_cols.get(consts.HOST)

                left_cols = {int(k): v for k, v in left_cols.items()}
                self.host_filter_result.add_left_cols(left_cols)
                LOGGER.debug("In Guest IV filter, host_select_cols: {}, host_left_cols: {}".format(host_select_cols,
                                                                                                   host_left_cols))
                # new_result = {}
                # for host_col_idx, _ in host_select_cols.items():
                #     host_col_idx = int(host_col_idx)
                #     is_left = left_cols.get(host_col_idx)
                #     new_result[host_col_idx] = is_left
                # self.host_left_cols = new_result
                # self._add_host_left_cols(self.host_left_cols)
                self._send_host_result_cols(consts.IV_VALUE_THRES)
                LOGGER.debug(
                    "[Result][FeatureSelection][Guest] Finish iv value threshold filter. Host left cols are: {}".format(
                        self.host_filter_result.get_left_cols()))

            else:
                iv_filter = feature_selection.IVValueSelectFilter(iv_param,
                                                                  self.filter_result.this_to_select_cols_index,
                                                                  self.binning_model)
                new_left_cols = iv_filter.fit(data_instances)
                # self._renew_final_left_cols(new_left_cols)
                self.filter_result.add_left_col_index(new_left_cols)

            LOGGER.debug(
                "[Result][FeatureSelection][Guest] Finish iv value threshold filter. Self left cols are: {}".format(
                    self.filter_result.get_left_cols()))
            self.iv_value_meta = iv_filter.get_meta_obj()
            self.results.append(iv_filter.get_param_obj())
            # self._renew_left_col_names()

        if method == consts.IV_PERCENTILE:

            iv_param = self.model_param.iv_percentile_param
            if self.local_only:
                iv_filter = feature_selection.IVPercentileFilter(iv_param,
                                                                 self.filter_result.this_to_select_cols_index,
                                                                 {},
                                                                 self.binning_model)
                new_left_cols = iv_filter.fit(data_instances)
                # self._renew_final_left_cols(new_left_cols)
                self.filter_result.add_left_col_index(new_left_cols)

            else:
                host_select_cols = self._get_host_select_cols(consts.IV_PERCENTILE)
                host_cols = {consts.HOST: host_select_cols}

                iv_filter = feature_selection.IVPercentileFilter(iv_param,
                                                                 self.filter_result.this_to_select_cols_index,
                                                                 host_cols,
                                                                 self.binning_model)
                new_left_cols = iv_filter.fit(data_instances)
                # self._renew_final_left_cols(new_left_cols)
                self.filter_result.add_left_col_index(new_left_cols)

                host_left_cols = iv_filter.host_cols
                # Only one host
                left_col_index = host_left_cols.get(consts.HOST)
                self.host_filter_result.add_left_cols(left_col_index)

                self._send_host_result_cols(consts.IV_PERCENTILE)
                LOGGER.info(
                    "[Result][FeatureSelection][Host]Finish iv percentile threshold filter. "
                    "Host left cols are: {}".format(self.host_filter_result.get_left_cols()))

            LOGGER.debug(
                "[Result][FeatureSelection][Guest]Finish iv percentile threshold filter. Self left cols are: {}".format(
                    self.filter_result.get_left_cols()))
            self.iv_percentile_meta = iv_filter.get_meta_obj()
            self.results.append(iv_filter.get_param_obj())
            # self._renew_left_col_names()

        if method == consts.COEFFICIENT_OF_VARIATION_VALUE_THRES:
            variance_coe_param = self.model_param.variance_coe_param
            coe_filter = feature_selection.CoeffOfVarValueFilter(variance_coe_param,
                                                                 self.filter_result.this_to_select_cols_index,
                                                                 self.static_obj)
            new_left_cols = coe_filter.fit(data_instances)
            # self._renew_final_left_cols(new_left_cols)
            self.filter_result.add_left_col_index(new_left_cols)
            self.static_obj = coe_filter.statics_obj

            LOGGER.info(
                "[Result][FeatureSelection][Guest]Finish coefficient of variance filter. Self left cols are: {}".format(
                    self.filter_result.get_left_cols()))

            self.variance_coe_meta = coe_filter.get_meta_obj()
            self.results.append(coe_filter.get_param_obj())
            # self._renew_left_col_names()

        if method == consts.UNIQUE_VALUE:
            unique_param = self.model_param.unique_param
            unique_filter = feature_selection.UniqueValueFilter(unique_param,
                                                                self.filter_result.this_to_select_cols_index,
                                                                self.static_obj)
            new_left_cols = unique_filter.fit(data_instances)
            # self._renew_final_left_cols(new_left_cols)
            self.filter_result.add_left_col_index(new_left_cols)

            self.static_obj = unique_filter.statics_obj
            self.unique_meta = unique_filter.get_meta_obj()
            self.results.append(unique_filter.get_param_obj())

            LOGGER.info(
                "[Result][FeatureSelection][Guest]Finish unique value filter. Current left cols are: {}".format(
                    self.filter_result.get_left_cols()))

        if method == consts.OUTLIER_COLS:
            outlier_param = self.model_param.outlier_param
            outlier_filter = feature_selection.OutlierFilter(outlier_param,
                                                             self.filter_result.this_to_select_cols_index)
            new_left_cols = outlier_filter.fit(data_instances)
            # self._renew_final_left_cols(new_left_cols)
            self.filter_result.add_left_col_index(new_left_cols)

            self.outlier_meta = outlier_filter.get_meta_obj()
            self.results.append(outlier_filter.get_param_obj())
            LOGGER.info(
                "[Result][FeatureSelection][Guest]Finish outlier filter. Self left cols are: {}".format(
                    self.filter_result.get_left_cols()))
            # self._renew_left_col_names()

    def _send_host_result_cols(self, filter_name):

        self.transfer_variable.result_left_cols.remote(self.host_filter_result.get_left_cols(),
                                                       role=consts.HOST,
                                                       idx=0,
                                                       suffix=(filter_name,))

        LOGGER.info(f"Sent result cols from guest to host, result cols are: {self.host_filter_result.get_left_cols()}")

    def _get_host_select_cols(self, filter_name):

        host_select_col_index = self.transfer_variable.host_select_cols.get(idx=0,
                                                                            suffix=(filter_name,))

        LOGGER.info("Received host_select_cols from host, host cols are: {}".format(host_select_col_index))

        self.host_filter_result.set_to_select_cols(host_select_col_index)
        # left_col_idx = set()
        # for col_idx, _ in host_select_cols.items():
        #     col_name = '.'.join(['host', str(col_idx)])
        #     left_col_idx.add(col_name)
        # self.host_cols_list.append(left_col_idx)

        return self.host_filter_result.to_select_cols_dict
