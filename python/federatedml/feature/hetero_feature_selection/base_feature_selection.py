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

import copy
import functools
import random

from federatedml.feature.feature_selection import filter_factory
from federatedml.feature.feature_selection.model_adapter.adapter_factory import adapter_factory
from federatedml.feature.feature_selection.selection_properties import SelectionProperties, CompletedSelectionResults
from federatedml.model_base import ModelBase
from federatedml.param.feature_selection_param import FeatureSelectionParam
from federatedml.protobuf.generated import feature_selection_param_pb2, feature_selection_meta_pb2
from federatedml.statistic.data_overview import get_header, \
    get_anonymous_header, look_up_names_from_header, header_alignment
from federatedml.transfer_variable.transfer_class.hetero_feature_selection_transfer_variable import \
    HeteroFeatureSelectionTransferVariable
from federatedml.util import LOGGER
from federatedml.util import abnormal_detection
from federatedml.util import consts
from federatedml.util.io_check import assert_io_num_rows_equal
from federatedml.util.schema_check import assert_schema_consistent

MODEL_PARAM_NAME = 'FeatureSelectionParam'
MODEL_META_NAME = 'FeatureSelectionMeta'
MODEL_NAME = 'HeteroFeatureSelection'


class BaseHeteroFeatureSelection(ModelBase):
    def __init__(self):
        super(BaseHeteroFeatureSelection, self).__init__()
        self.transfer_variable = HeteroFeatureSelectionTransferVariable()

        self.curt_select_properties = SelectionProperties()
        self.completed_selection_result = CompletedSelectionResults()
        self.loaded_local_select_properties = dict()
        self.loaded_host_filter_results = dict()

        self.schema = None
        self.header = None
        self.anonymous_header = None
        self.party_name = 'Base'
        # Possible previous model
        self.binning_model = None
        self.static_obj = None
        self.model_param = FeatureSelectionParam()
        # self.meta_dicts = {}
        self.meta_list = []
        self.isometric_models = {}

    def _init_model(self, params):
        self.model_param = params
        # self.cols_index = params.select_cols
        self.filter_methods = params.filter_methods
        # self.local_only = params.local_only

    def _init_select_params(self, data_instances):
        if self.schema is None:
            self.schema = data_instances.schema

        if self.header is not None:
            # load current data anonymous header for prediction with model of version < 1.9.0
            # if len(self.completed_selection_result.anonymous_header) == 0:
            if self.anonymous_header is None:
                data_anonymous_header = get_anonymous_header(data_instances)
                # LOGGER.info(f"data_anonymous_header: {data_anonymous_header}")
                self.anonymous_header = data_anonymous_header
                self.completed_selection_result.set_anonymous_header(data_anonymous_header)
                if self.role == consts.HOST:
                    anonymous_header_in_old_format = self.anonymous_generator. \
                        generated_compatible_anonymous_header_with_old_version(data_anonymous_header)
                    anonymous_dict = dict(zip(anonymous_header_in_old_format, data_anonymous_header))
                    self.transfer_variable.host_anonymous_header_dict.remote(anonymous_dict,
                                                                             role=consts.GUEST,
                                                                             idx=0)

                    for filter_name, select_properties in self.loaded_local_select_properties.items():
                        self.completed_selection_result.add_filter_results(filter_name, select_properties)
                else:
                    host_anonymous_dict_list = self.transfer_variable.host_anonymous_header_dict.get(idx=-1)
                    for filter_name, cur_select_properties in self.loaded_local_select_properties.items():
                        cur_host_select_properties_list = []
                        host_feature_values_obj_list, host_left_cols_obj_list = self.loaded_host_filter_results[
                            filter_name]
                        for i, host_left_cols_obj in enumerate(host_left_cols_obj_list):
                            cur_host_select_properties = SelectionProperties()
                            old_host_header = list(host_anonymous_dict_list[i].keys())
                            host_feature_values = host_feature_values_obj_list[i].feature_values
                            cur_host_select_properties.load_properties_with_new_header(old_host_header,
                                                                                       host_feature_values,
                                                                                       host_left_cols_obj,
                                                                                       host_anonymous_dict_list[i])
                            cur_host_select_properties_list.append(cur_host_select_properties)

                        self.completed_selection_result.add_filter_results(filter_name,
                                                                           cur_select_properties,
                                                                           cur_host_select_properties_list)
            return
        self.schema = data_instances.schema
        header = get_header(data_instances)
        anonymous_header = get_anonymous_header(data_instances)
        self.header = header
        self.anonymous_header = anonymous_header
        self.curt_select_properties.set_header(header)
        # use anonymous header of input data
        self.curt_select_properties.set_anonymous_header(anonymous_header)
        self.curt_select_properties.set_last_left_col_indexes([x for x in range(len(header))])
        if self.model_param.select_col_indexes == -1:
            self.curt_select_properties.set_select_all_cols()
        else:
            self.curt_select_properties.add_select_col_indexes(self.model_param.select_col_indexes)
        if self.model_param.use_anonymous:
            select_names = look_up_names_from_header(self.model_param.select_names, anonymous_header, header)
            # LOGGER.debug(f"use_anonymous is true, select names: {select_names}")
        else:
            select_names = self.model_param.select_names
        self.curt_select_properties.add_select_col_names(select_names)
        self.completed_selection_result.set_header(header)
        self.completed_selection_result.set_anonymous_header(anonymous_header)
        self.completed_selection_result.set_select_col_names(self.curt_select_properties.select_col_names)
        self.completed_selection_result.set_all_left_col_indexes(self.curt_select_properties.all_left_col_indexes)

    def _get_meta(self):
        meta_dicts = {'filter_methods': self.filter_methods,
                      'cols': self.completed_selection_result.get_select_col_names(),
                      'need_run': self.need_run,
                      "filter_metas": self.meta_list}
        meta_protobuf_obj = feature_selection_meta_pb2.FeatureSelectionMeta(**meta_dicts)
        return meta_protobuf_obj

    def _get_param(self):
        # LOGGER.debug("curt_select_properties.left_col_name: {}, completed_selection_result: {}".format(
        #     self.curt_select_properties.left_col_names, self.completed_selection_result.all_left_col_names
        # ))
        # LOGGER.debug("Length of left cols: {}".format(len(self.completed_selection_result.all_left_col_names)))
        # left_cols = {x: True for x in self.curt_select_properties.left_col_names}
        left_cols = {x: True for x in self.completed_selection_result.all_left_col_names}
        final_left_cols = feature_selection_param_pb2.LeftCols(
            original_cols=self.completed_selection_result.get_select_col_names(),
            left_cols=left_cols
        )

        host_col_names = []
        if self.role == consts.GUEST:
            for host_id, this_host_name in enumerate(self.completed_selection_result.get_host_sorted_col_names()):
                party_id = self.component_properties.host_party_idlist[host_id]
                # LOGGER.debug("In _get_param, this_host_name: {}, party_id: {}".format(this_host_name, party_id))

                host_col_names.append(feature_selection_param_pb2.HostColNames(col_names=this_host_name,
                                                                               party_id=str(party_id)))
        else:
            party_id = self.component_properties.local_partyid
            # if self.anonymous_header:
            # anonymous_names = self.anonymous_header
            """else:
                anonymous_names = [anonymous_generator.generate_anonymous(fid, model=self)
                                   for fid in range(len(self.header))]
            """
            host_col_names.append(feature_selection_param_pb2.HostColNames(col_names=self.anonymous_header,
                                                                           party_id=str(party_id)))
        col_name_to_anonym_dict = None
        if self.header and self.anonymous_header:
            col_name_to_anonym_dict = dict(zip(self.header, self.anonymous_header))

        result_obj = feature_selection_param_pb2.FeatureSelectionParam(
            results=self.completed_selection_result.filter_results,
            final_left_cols=final_left_cols,
            col_names=self.completed_selection_result.get_sorted_col_names(),
            host_col_names=host_col_names,
            header=self.curt_select_properties.header,
            col_name_to_anonym_dict=col_name_to_anonym_dict
        )
        return result_obj

    def save_data(self):
        return self.data_output

    def export_model(self):
        # LOGGER.debug("Model output is : {}".format(self.model_output))
        """
        if self.model_output is not None:
            LOGGER.debug("model output already exists, return directly")
            return self.model_output
        """

        meta_obj = self._get_meta()
        param_obj = self._get_param()
        result = {
            MODEL_META_NAME: meta_obj,
            MODEL_PARAM_NAME: param_obj
        }
        self.model_output = result
        return result

    def _load_selection_model(self, model_dict):
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
        # LOGGER.info(f"col_name_to_anonym_dict: {model_param.col_name_to_anonym_dict}")
        self.header = header
        self.curt_select_properties.set_header(header)
        self.completed_selection_result.set_header(header)
        self.curt_select_properties.set_last_left_col_indexes([x for x in range(len(header))])
        self.curt_select_properties.add_select_col_names(header)

        # for model ver >= 1.9.0
        if model_param.col_name_to_anonym_dict:
            col_name_to_anonym_dict = dict(model_param.col_name_to_anonym_dict)
            self.anonymous_header = [col_name_to_anonym_dict[x] for x in header]
            self.completed_selection_result.set_anonymous_header(self.anonymous_header)

            host_col_names_list = model_param.host_col_names
            for result in model_param.results:
                cur_select_properties = copy.deepcopy(self.curt_select_properties)
                feature_values, left_cols_obj = dict(result.feature_values), result.left_cols
                cur_select_properties.load_properties(header, feature_values, left_cols_obj)

                cur_host_select_properties_list = []
                host_feature_values_obj_list = list(result.host_feature_values)
                host_left_cols_obj_list = list(result.host_left_cols)
                for i, host_left_cols_obj in enumerate(host_left_cols_obj_list):
                    cur_host_select_properties = SelectionProperties()
                    host_col_names_obj = host_col_names_list[i]
                    host_header = list(host_col_names_obj.col_names)
                    host_feature_values = host_feature_values_obj_list[i].feature_values
                    cur_host_select_properties.load_properties(host_header, host_feature_values, host_left_cols_obj)
                    cur_host_select_properties_list.append(cur_host_select_properties)

                self.completed_selection_result.add_filter_results(result.filter_name,
                                                                   cur_select_properties,
                                                                   cur_host_select_properties_list)
        # for model ver 1.8.x
        else:
            LOGGER.warning(f"Anonymous column name dictionary not found in given model."
                           f"Will infer host(s)' anonymous names.")
            """
            self.loaded_host_col_names_list = [list(host_col_names_obj.col_names)
                                               for host_col_names_obj in model_param.host_col_names]
            """
            for result in model_param.results:
                cur_select_properties = copy.deepcopy(self.curt_select_properties)
                feature_values, left_cols_obj = dict(result.feature_values), result.left_cols
                cur_select_properties.load_properties(header, feature_values, left_cols_obj)
                # record local select properties
                self.loaded_local_select_properties[result.filter_name] = cur_select_properties

                host_feature_values_obj_list = list(result.host_feature_values)
                host_left_cols_obj_list = list(result.host_left_cols)
                self.loaded_host_filter_results[result.filter_name] = (host_feature_values_obj_list,
                                                                       host_left_cols_obj_list)

        final_left_cols_names = dict(model_param.final_left_cols.left_cols)
        # LOGGER.debug("final_left_cols_names: {}".format(final_left_cols_names))
        for col_name, _ in final_left_cols_names.items():
            self.curt_select_properties.add_left_col_name(col_name)
        self.completed_selection_result.add_filter_results(filter_name='conclusion',
                                                           select_properties=self.curt_select_properties)
        self.update_curt_select_param()

    def _load_isometric_model(self, iso_model):
        LOGGER.debug(f"When loading isometric_model, iso_model names are:"
                     f" {iso_model.keys()}")
        for cpn_name, model_dict in iso_model.items():
            model_param = None
            model_meta = None
            for name, model_pb in model_dict.items():
                if name.endswith("Param"):
                    model_param = model_pb
                else:
                    model_meta = model_pb
            model_name = model_param.model_name
            if model_name in self.isometric_models:
                raise ValueError("Should not load two same type isometric models"
                                 " in feature selection")
            adapter = adapter_factory(model_name)
            this_iso_model = adapter.convert(model_meta, model_param)
            self.isometric_models[model_name] = this_iso_model

    def load_model(self, model_dict):
        LOGGER.debug(f"Start to load model")
        if 'model' in model_dict:
            LOGGER.debug("Loading selection model")
            self._load_selection_model(model_dict)

        if 'isometric_model' in model_dict:
            LOGGER.debug("Loading isometric_model")
            self._load_isometric_model(model_dict['isometric_model'])

    @staticmethod
    def select_cols(instance, left_col_idx):
        instance.features = instance.features[left_col_idx]
        return instance

    def _transfer_data(self, data_instances):
        f = functools.partial(self.select_cols,
                              left_col_idx=self.completed_selection_result.all_left_col_indexes)

        new_data = data_instances.mapValues(f)

        # LOGGER.debug("When transfering, all left_col_names: {}".format(
        #    self.completed_selection_result.all_left_col_names
        # ))

        new_data = self.set_schema(new_data,
                                   self.completed_selection_result.all_left_col_names,
                                   self.completed_selection_result.all_left_anonymous_col_names)

        # one_data = new_data.first()[1]
        # LOGGER.debug(
        #    "In feature selection transform, Before transform: {}, length: {} After transform: {}, length: {}".format(
        #        before_one_data[1].features, len(before_one_data[1].features),
        #        one_data.features, len(one_data.features)))

        return new_data

    def _abnormal_detection(self, data_instances):
        """
        Make sure input data_instances is valid.
        """
        abnormal_detection.empty_table_detection(data_instances)
        abnormal_detection.empty_feature_detection(data_instances)
        self.check_schema_content(data_instances.schema)

    def set_schema(self, data_instance, header=None, anonymous_header=None):
        if header is None:
            self.schema["header"] = self.curt_select_properties.header
            self.schema["anonymous_header"] = self.curt_select_properties.anonymous_header
        else:
            self.schema["header"] = header
            self.schema["anonymous_header"] = anonymous_header
        data_instance.schema = self.schema
        return data_instance

    def update_curt_select_param(self):
        new_select_properties = SelectionProperties()
        # all select properties must have the same header
        new_select_properties.set_header(self.curt_select_properties.header)
        new_select_properties.set_anonymous_header(self.curt_select_properties.anonymous_header)
        new_select_properties.set_last_left_col_indexes(self.curt_select_properties.all_left_col_indexes)
        new_select_properties.add_select_col_names(self.curt_select_properties.left_col_names)
        self.curt_select_properties = new_select_properties

    def _filter(self, data_instances, method, suffix, idx=0):
        this_filter = filter_factory.get_filter(filter_name=method, model_param=self.model_param,
                                                role=self.role, model=self, idx=idx)
        if method == consts.STATISTIC_FILTER:
            method = self.model_param.statistic_param.metrics[idx]
        elif method == consts.IV_FILTER:
            metric = self.model_param.iv_param.metrics[idx]
            f_type = self.model_param.iv_param.filter_type[idx]
            method = f"{metric}_{f_type}"
        elif method == consts.PSI_FILTER:
            metric = self.model_param.psi_param.metrics[idx]
            f_type = self.model_param.psi_param.filter_type[idx]
            method = f"{metric}_{f_type}"
        this_filter.set_selection_properties(self.curt_select_properties)

        this_filter.set_transfer_variable(self.transfer_variable)
        # .info(f"this_filter type: {this_filter.filter_type}, method: {method}, filter obj: {this_filter}")
        self.curt_select_properties = this_filter.fit(data_instances, suffix).selection_properties
        # LOGGER.info(f"filter.fit called")
        host_select_properties = getattr(this_filter, 'host_selection_properties', None)
        # if host_select_properties is not None:
        #     LOGGER.debug("method: {}, host_select_properties: {}".format(
        #         method, host_select_properties[0].all_left_col_names))

        self.completed_selection_result.add_filter_results(filter_name=method,
                                                           select_properties=self.curt_select_properties,
                                                           host_select_properties=host_select_properties)
        last_col_nums = len(self.curt_select_properties.last_left_col_names)
        left_col_names = self.curt_select_properties.left_col_names
        self.add_summary(method, {
            "last_col_nums": last_col_nums,
            "left_col_nums": len(left_col_names),
            "left_col_names": left_col_names
        })
        # LOGGER.debug("method: {}, selection_cols: {}, left_cols: {}".format(
        #     method, self.curt_select_properties.select_col_names, self.curt_select_properties.left_col_names))
        self.update_curt_select_param()
        # LOGGER.debug("After updated, method: {}, selection_cols: {}".format(
        #     method, self.curt_select_properties.select_col_names))
        self.meta_list.append(this_filter.get_meta_obj())

    def fit(self, data_instances):
        LOGGER.info("Start Hetero Selection Fit and transform.")
        self._abnormal_detection(data_instances)
        self._init_select_params(data_instances)

        original_col_nums = len(self.curt_select_properties.last_left_col_names)

        empty_cols = False
        if len(self.curt_select_properties.select_col_indexes) == 0:
            LOGGER.warning("None of columns has been set to select, "
                           "will randomly select one column to participate in fitting filter(s). "
                           "All columns will be kept, "
                           "but be aware that this may lead to unexpected behavior.")
            header = data_instances.schema.get("header")
            select_idx = random.choice(range(len(header)))
            self.curt_select_properties.select_col_indexes = [select_idx]
            self.curt_select_properties.select_col_names = [header[select_idx]]
            empty_cols = True
        suffix = self.filter_methods
        if self.role == consts.HOST:
            self.transfer_variable.host_empty_cols.remote(empty_cols, role=consts.GUEST, idx=0, suffix=suffix)
        else:
            host_empty_cols_list = self.transfer_variable.host_empty_cols.get(idx=-1, suffix=suffix)
            host_list = self.component_properties.host_party_idlist
            for idx, res in enumerate(host_empty_cols_list):
                if res:
                    LOGGER.warning(f"Host {host_list[idx]}'s select columns are empty;"
                                   f"host {host_list[idx]} will randomly select one "
                                   f"column to participate in fitting filter(s). "
                                   f"All columns from this host will be kept, "
                                   f"but be aware that this may lead to unexpected behavior.")
        for filter_idx, method in enumerate(self.filter_methods):
            if method in [consts.STATISTIC_FILTER, consts.IV_FILTER, consts.PSI_FILTER,
                          consts.HETERO_SBT_FILTER, consts.HOMO_SBT_FILTER, consts.HETERO_FAST_SBT_FILTER,
                          consts.VIF_FILTER]:
                if method == consts.STATISTIC_FILTER:
                    metrics = self.model_param.statistic_param.metrics
                elif method == consts.IV_FILTER:
                    metrics = self.model_param.iv_param.metrics
                elif method == consts.PSI_FILTER:
                    metrics = self.model_param.psi_param.metrics
                elif method in [consts.HETERO_SBT_FILTER, consts.HOMO_SBT_FILTER, consts.HETERO_FAST_SBT_FILTER]:
                    metrics = self.model_param.sbt_param.metrics
                elif method == consts.VIF_FILTER:
                    metrics = self.model_param.vif_param.metrics
                else:
                    raise ValueError(f"method: {method} is not supported")
                for idx, _ in enumerate(metrics):
                    self._filter(data_instances, method,
                                 suffix=(str(filter_idx), str(idx)), idx=idx)
            else:
                self._filter(data_instances, method, suffix=str(filter_idx))
        last_col_nums = self.curt_select_properties.last_left_col_names

        self.add_summary("all", {
            "last_col_nums": original_col_nums,
            "left_col_nums": len(last_col_nums),
            "left_col_names": last_col_nums
        })

        new_data = self._transfer_data(data_instances)
        # LOGGER.debug(f"Final summary: {self.summary()}")
        LOGGER.info("Finish Hetero Selection Fit and transform.")
        return new_data

    @assert_io_num_rows_equal
    @assert_schema_consistent
    def transform(self, data_instances):
        self._abnormal_detection(data_instances)
        self._init_select_params(data_instances)
        # align data instance to model header & anonymous header
        data_instances = header_alignment(data_instances, self.header, self.anonymous_header)
        new_data = self._transfer_data(data_instances)
        return new_data
