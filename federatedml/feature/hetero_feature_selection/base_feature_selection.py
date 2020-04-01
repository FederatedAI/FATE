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

import functools

from google.protobuf import json_format

from arch.api.utils import log_utils
from federatedml.feature.feature_selection import filter_factory
from federatedml.feature.feature_selection.selection_properties import SelectionProperties, CompletedSelectionResults
from federatedml.feature.hetero_feature_binning.hetero_binning_guest import HeteroFeatureBinningGuest
from federatedml.feature.hetero_feature_binning.hetero_binning_host import HeteroFeatureBinningHost
from federatedml.model_base import ModelBase
from federatedml.param.feature_selection_param import FeatureSelectionParam
from federatedml.protobuf.generated import feature_selection_param_pb2, feature_selection_meta_pb2
from federatedml.statistic.data_overview import get_header
from federatedml.transfer_variable.transfer_class.hetero_feature_selection_transfer_variable import \
    HeteroFeatureSelectionTransferVariable
from federatedml.util import abnormal_detection
from federatedml.util import consts

LOGGER = log_utils.getLogger()

MODEL_PARAM_NAME = 'FeatureSelectionParam'
MODEL_META_NAME = 'FeatureSelectionMeta'
MODEL_NAME = 'HeteroFeatureSelection'


class BaseHeteroFeatureSelection(ModelBase):
    def __init__(self):
        super(BaseHeteroFeatureSelection, self).__init__()
        self.transfer_variable = HeteroFeatureSelectionTransferVariable()

        self.curt_select_properties = SelectionProperties()
        self.completed_selection_result = CompletedSelectionResults()

        self.schema = None
        self.party_name = 'Base'
        # Possible previous model
        self.binning_model = None
        self.static_obj = None
        self.model_param = FeatureSelectionParam()
        self.meta_dicts = {}

    def _init_model(self, params):
        self.model_param = params
        # self.cols_index = params.select_cols
        self.filter_methods = params.filter_methods
        # self.local_only = params.local_only

    def _init_select_params(self, data_instances):
        if self.schema is not None:
            return
        self.schema = data_instances.schema
        header = get_header(data_instances)
        self.curt_select_properties.set_header(header)
        self.curt_select_properties.set_last_left_col_indexes([x for x in range(len(header))])
        if self.model_param.select_col_indexes == -1:
            self.curt_select_properties.set_select_all_cols()
        else:
            self.curt_select_properties.add_select_col_indexes(self.model_param.select_col_indexes)
        self.curt_select_properties.add_select_col_names(self.model_param.select_names)
        self.completed_selection_result.set_header(header)
        self.completed_selection_result.set_select_col_names(self.curt_select_properties.select_col_names)
        self.completed_selection_result.set_all_left_col_indexes(self.curt_select_properties.all_left_col_indexes)

    def _get_meta(self):
        self.meta_dicts['filter_methods'] = self.filter_methods
        self.meta_dicts['cols'] = self.completed_selection_result.get_select_col_names()
        self.meta_dicts['need_run'] = self.need_run
        meta_protobuf_obj = feature_selection_meta_pb2.FeatureSelectionMeta(**self.meta_dicts)
        return meta_protobuf_obj

    def _get_param(self):
        LOGGER.debug("curt_select_properties.left_col_name: {}, completed_selection_result: {}".format(
            self.curt_select_properties.left_col_names, self.completed_selection_result.all_left_col_names
        ))
        LOGGER.debug("Length of left cols: {}".format(len(self.completed_selection_result.all_left_col_names)))
        # left_cols = {x: True for x in self.curt_select_properties.left_col_names}
        left_cols = {x: True for x in self.completed_selection_result.all_left_col_names}
        final_left_cols = feature_selection_param_pb2.LeftCols(
            original_cols=self.completed_selection_result.get_select_col_names(),
            left_cols=left_cols
        )

        host_col_names = []
        for host_id, this_host_name in enumerate(self.completed_selection_result.get_host_sorted_col_names()):
            party_id = self.component_properties.host_party_idlist[host_id]
            LOGGER.debug("In _get_param, this_host_name: {}, party_id: {}".format(this_host_name, party_id))

            host_col_names.append(feature_selection_param_pb2.HostColNames(col_names=this_host_name,
                                                                           party_id=str(party_id)))

        result_obj = feature_selection_param_pb2.FeatureSelectionParam(
            results=self.completed_selection_result.filter_results,
            final_left_cols=final_left_cols,
            col_names=self.completed_selection_result.get_sorted_col_names(),
            host_col_names=host_col_names,
            header=self.curt_select_properties.header
        )

        json_result = json_format.MessageToJson(result_obj)
        LOGGER.debug("json_result: {}".format(json_result))
        return result_obj

    def save_data(self):
        return self.data_output

    def export_model(self):
        LOGGER.debug("Model output is : {}".format(self.model_output))
        if self.model_output is not None:
            LOGGER.debug("model output is already exist, return directly")
            return self.model_output

        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            MODEL_META_NAME: meta_obj,
            MODEL_PARAM_NAME: param_obj
        }
        self.model_output = result
        return result

    def load_model(self, model_dict):

        if 'model' in model_dict:
            # self._parse_need_run(model_dict, MODEL_META_NAME)
            LOGGER.debug("Feature selection need run: {}".format(self.need_run))
            if not self.need_run:
                return
            model_param = list(model_dict.get('model').values())[0].get(MODEL_PARAM_NAME)
            model_meta = list(model_dict.get('model').values())[0].get(MODEL_META_NAME)

            self.model_output = {
                MODEL_META_NAME: model_meta,
                MODEL_PARAM_NAME: model_param
            }

            header = list(model_param.header)
            self.curt_select_properties.set_header(header)
            self.completed_selection_result.set_header(header)
            self.curt_select_properties.set_last_left_col_indexes([x for x in range(len(header))])
            self.curt_select_properties.add_select_col_names(header)

            final_left_cols_names = dict(model_param.final_left_cols.left_cols)
            LOGGER.debug("final_left_cols_names: {}".format(final_left_cols_names))
            for col_name, _ in final_left_cols_names.items():
                self.curt_select_properties.add_left_col_name(col_name)
            self.completed_selection_result.add_filter_results(filter_name='conclusion',
                                                               select_properties=self.curt_select_properties)
            self.update_curt_select_param()
            LOGGER.debug("After load model, completed_selection_result.all_left_col_indexes: {}".format(
                self.completed_selection_result.all_left_col_indexes))

        if 'isometric_model' in model_dict:

            LOGGER.debug("Has isometric_model, model_dict: {}".format(model_dict))
            if self.party_name == consts.GUEST:
                self.binning_model = HeteroFeatureBinningGuest()
            else:
                self.binning_model = HeteroFeatureBinningHost()

            new_model_dict = {'model': model_dict['isometric_model']}
            self.binning_model.load_model(new_model_dict)

    @staticmethod
    def select_cols(instance, left_col_idx):
        instance.features = instance.features[left_col_idx]
        return instance

    def _transfer_data(self, data_instances):

        before_one_data = data_instances.first()
        f = functools.partial(self.select_cols,
                              left_col_idx=self.completed_selection_result.all_left_col_indexes)

        new_data = data_instances.mapValues(f)

        new_data = self.set_schema(new_data, self.completed_selection_result.all_left_col_names)

        one_data = new_data.first()[1]
        LOGGER.debug(
            "In feature selection transform, Before transform: {}, length: {} After transform: {}, length: {}".format(
                before_one_data[1].features, len(before_one_data[1].features),
                one_data.features, len(one_data.features)))

        return new_data

    def _abnormal_detection(self, data_instances):
        """
        Make sure input data_instances is valid.
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)

    def set_schema(self, data_instance, header=None):
        if header is None:
            self.schema["header"] = self.curt_select_properties.header
        else:
            self.schema["header"] = header
        data_instance.schema = self.schema
        return data_instance

    def update_curt_select_param(self):
        new_select_properties = SelectionProperties()
        new_select_properties.set_header(self.curt_select_properties.header)
        new_select_properties.set_last_left_col_indexes(self.curt_select_properties.all_left_col_indexes)
        new_select_properties.add_select_col_names(self.curt_select_properties.left_col_names)
        LOGGER.debug("In update_curt_select_param, header: {}, cols_map: {},"
                     "last_left_col_indexes: {}, select_col_names: {}".format(
            new_select_properties.header,
            new_select_properties.col_name_maps,
            new_select_properties.last_left_col_indexes,
            new_select_properties.select_col_names
        ))
        self.curt_select_properties = new_select_properties

    def _filter(self, data_instances, method, suffix):
        this_filter = filter_factory.get_filter(filter_name=method, model_param=self.model_param, role=self.role)
        this_filter.set_selection_properties(self.curt_select_properties)
        this_filter.set_statics_obj(self.static_obj)
        this_filter.set_binning_obj(self.binning_model)
        this_filter.set_transfer_variable(self.transfer_variable)
        self.curt_select_properties = this_filter.fit(data_instances, suffix).selection_properties
        host_select_properties = getattr(this_filter, 'host_selection_properties', None)
        LOGGER.debug("method: {}, host_select_properties: {}".format(
            method, host_select_properties))

        self.completed_selection_result.add_filter_results(filter_name=method,
                                                           select_properties=self.curt_select_properties,
                                                           host_select_properties=host_select_properties)
        LOGGER.debug("method: {}, selection_cols: {}, left_cols: {}".format(
            method, self.curt_select_properties.select_col_names, self.curt_select_properties.left_col_names))
        self.update_curt_select_param()
        LOGGER.debug("After updated, method: {}, selection_cols: {}, left_cols: {}".format(
            method, self.curt_select_properties.select_col_names, self.curt_select_properties.left_col_names))
        self.meta_dicts = this_filter.get_meta_obj(self.meta_dicts)

    def fit(self, data_instances):
        LOGGER.info("Start Hetero Selection Fit and transform.")
        self._abnormal_detection(data_instances)
        self._init_select_params(data_instances)

        if len(self.curt_select_properties.select_col_indexes) == 0:
            LOGGER.warning("None of columns has been set to select")
        else:
            for filter_idx, method in enumerate(self.filter_methods):
                self._filter(data_instances, method, suffix=str(filter_idx))

        new_data = self._transfer_data(data_instances)
        LOGGER.info("Finish Hetero Selection Fit and transform.")
        return new_data

    def transform(self, data_instances):
        self._abnormal_detection(data_instances)
        self._init_select_params(data_instances)
        new_data = self._transfer_data(data_instances)
        return new_data
