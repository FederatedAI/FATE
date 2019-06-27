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


from arch.api import federation
from arch.api.utils import log_utils
from federatedml.feature import feature_selection
from federatedml.feature.hetero_feature_selection.base_feature_selection import BaseHeteroFeatureSelection
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroFeatureSelectionGuest(BaseHeteroFeatureSelection):
    def __init__(self):
        super(HeteroFeatureSelectionGuest, self).__init__()
        self.host_left_cols = {}
        self.static_obj = None
        self.binning_model = None
        self.flowid = ''
        self.party_name = consts.GUEST

    def fit(self, data_instances):
        self._abnormal_detection(data_instances)
        self._init_cols(data_instances)

        for method in self.filter_method:
            self.filter_one_method(data_instances, method)
            print("After method: {}, left_cols: {}".format(method, self.left_cols))
            # self._renew_left_col_names()

        new_data = self._transfer_data(data_instances)
        return new_data

    def transform(self, data_instances):
        self._abnormal_detection(data_instances)

        # self._init_cols(data_instances)
        LOGGER.info("[Result][FeatureSelection][Guest]In transform, Self left cols are: {}".format(
            self.left_cols
        ))
        new_data = self._transfer_data(data_instances)
        self._reset_header()
        new_data.schema['header'] = self.header

        return new_data

    def filter_one_method(self, data_instances, method):

        if method == consts.IV_VALUE_THRES:
            iv_param = self.model_param.iv_value_param
            iv_filter = feature_selection.IVValueSelectFilter(iv_param,
                                                              self.cols,
                                                              self.binning_model)
            new_left_cols = iv_filter.fit(data_instances)

            # self.left_cols = new_left_cols
            self._renew_final_left_cols(new_left_cols)

            if not self.local_only:
                host_select_cols = self._get_host_select_cols(consts.IV_VALUE_THRES)
                host_left_cols = iv_filter.host_cols
                # Only one host
                left_cols = host_left_cols.get(consts.HOST)
                new_result = {}
                for host_col_name in host_select_cols:
                    if host_col_name not in left_cols:
                        LOGGER.warning(
                            "Host column {} has not been set in feature binning module".format(host_col_name))
                        continue
                    is_left = left_cols.get(host_col_name)
                    new_result[host_col_name] = is_left
                self.host_left_cols = new_result
                self._send_host_result_cols(consts.IV_VALUE_THRES)
                LOGGER.info(
                    "[Result][FeatureSelection][Guest] Finish iv value threshold filter. Host left cols are: {}".format(
                        self.host_left_cols))

            iv_filter.display_feature_result(self.party_name)
            LOGGER.info(
                "[Result][FeatureSelection][Guest] Finish iv value threshold filter. Self left cols are: {}".format(
                    self.left_cols))
            self.iv_value_meta = iv_filter.get_meta_obj()
            self.results.append(iv_filter.get_param_obj())
            # self._renew_left_col_names()

        if method == consts.IV_PERCENTILE:

            iv_param = self.model_param.iv_percentile_param
            if self.local_only:
                iv_filter = feature_selection.IVPercentileFilter(iv_param,
                                                                 self.cols,
                                                                 {},
                                                                 self.binning_model)
                new_left_cols = iv_filter.fit(data_instances)
                self._renew_final_left_cols(new_left_cols)


            else:
                host_select_cols = self._get_host_select_cols(consts.IV_PERCENTILE)
                host_cols = {consts.HOST: host_select_cols}
                iv_filter = feature_selection.IVPercentileFilter(iv_param,
                                                                 self.cols,
                                                                 host_cols,
                                                                 self.binning_model)
                new_left_cols = iv_filter.fit(data_instances)
                self._renew_final_left_cols(new_left_cols)

                host_left_cols = iv_filter.host_cols
                # Only one host
                left_cols = host_left_cols.get(consts.HOST)
                new_result = {}
                for host_col_name in host_select_cols:
                    if host_col_name not in left_cols:
                        LOGGER.warning(
                            "Host column {} has not been set in feature binning module".format(host_col_name))
                        continue
                    is_left = left_cols.get(host_col_name)
                    new_result[host_col_name] = is_left
                self.host_left_cols = new_result
                self._send_host_result_cols(consts.IV_PERCENTILE)
                LOGGER.info(
                    "[Result][FeatureSelection][Host]Finish iv value threshold filter. Host left cols are: {}".format(
                        self.host_left_cols))

            iv_filter.display_feature_result(self.party_name)
            LOGGER.info(
                "[Result][FeatureSelection][Guest]Finish iv value threshold filter. Self left cols are: {}".format(
                    self.left_cols))
            self.iv_percentile_meta = iv_filter.get_meta_obj()
            self.results.append(iv_filter.get_param_obj())
            # self._renew_left_col_names()

        if method == consts.COEFFICIENT_OF_VARIATION_VALUE_THRES:
            coe_param = self.model_param.coe_param
            coe_filter = feature_selection.CoeffOfVarValueFilter(coe_param, self.cols, self.static_obj)
            new_left_cols = coe_filter.fit(data_instances)
            self._renew_final_left_cols(new_left_cols)

            self.static_obj = coe_filter.statics_obj
            self.coe_meta = coe_filter.get_meta_obj()
            self.results.append(coe_filter.get_param_obj())
            # self._renew_left_col_names()
            coe_filter.display_feature_result(self.party_name)
            LOGGER.info(
                "[Result][FeatureSelection][Guest] Finish coefficient threshold filter. Self left cols are: {}".format(
                    self.left_cols))

        if method == consts.UNIQUE_VALUE:

            unique_param = self.model_param.unique_param
            unique_filter = feature_selection.UniqueValueFilter(unique_param, self.cols, self.static_obj)
            new_left_cols = unique_filter.fit(data_instances)
            self._renew_final_left_cols(new_left_cols)

            self.static_obj = unique_filter.statics_obj
            self.unique_meta = unique_filter.get_meta_obj()
            self.results.append(unique_filter.get_param_obj())
            unique_filter.display_feature_result(self.party_name)

            LOGGER.info("[Result][FeatureSelection][Guest]Finish unique value filter. Current left cols are: {}".format(
                self.left_cols))

        if method == consts.OUTLIER_COLS:
            outlier_param = self.model_param.outlier_param
            outlier_filter = feature_selection.OutlierFilter(outlier_param, self.cols)
            new_left_cols = outlier_filter.fit(data_instances)
            self._renew_final_left_cols(new_left_cols)

            self.outlier_meta = outlier_filter.get_meta_obj()
            self.results.append(outlier_filter.get_param_obj())
            # self._renew_left_col_names()
            outlier_filter.display_feature_result(self.party_name)
            LOGGER.info("[Result][FeatureSelection][Guest]Finish outlier cols filter. Current left cols are: {}".format(
                self.left_cols))

    def _send_host_result_cols(self, filter_name):
        result_cols_id = self.transfer_variable.generate_transferid(self.transfer_variable.result_left_cols,
                                                                    filter_name)
        federation.remote(self.host_left_cols,
                          name=self.transfer_variable.result_left_cols.name,
                          tag=result_cols_id,
                          role=consts.HOST,
                          idx=0)
        LOGGER.info("Sent result cols from guest to host, result cols are: {}".format(self.host_left_cols))

    def _get_host_select_cols(self, filter_name):
        host_select_cols_id = self.transfer_variable.generate_transferid(self.transfer_variable.host_select_cols,
                                                                         filter_name)
        host_select_cols = federation.get(name=self.transfer_variable.host_select_cols.name,
                                          tag=host_select_cols_id,
                                          idx=0)
        LOGGER.info("Received host_select_cols from host, host cols are: {}".format(host_select_cols))
        return host_select_cols
