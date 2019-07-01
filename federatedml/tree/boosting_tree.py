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
################################################################################
#
#
################################################################################

# =============================================================================
# Boostring Tree
# =============================================================================
import numpy as np
from fate_flow.manager.tracking import Tracking 
from fate_flow.entity.metric import Metric
from fate_flow.entity.metric import MetricMeta
from federatedml.param.boosting_tree_param import BoostingTreeParam
from federatedml.model_selection.KFold import KFold
from federatedml.util import abnormal_detection
from federatedml.util import consts
from federatedml.feature.sparse_vector import SparseVector
from federatedml.model_base import ModelBase


class BoostingTree(ModelBase):
    def __init__(self):
        super(BoostingTree, self).__init__()
        self.tree_param = None
        self.task_type=None
        self.objective_param = None
        self.learning_rate = None
        self.learning_rate = None
        self.num_trees = None
        self.subsample_feature_rate = None
        self.n_iter_no_change = None
        self.encrypt_param = None
        self.tol = 0.0
        self.quantile_method = None
        self.bin_num = None
        self.bin_gap = None
        self.bin_sample_num = None
        self.calculated_mode = None
        self.re_encrypted_rate = None
        self.predict_param = None
        self.cv_param = None
        self.feature_name_fid_mapping = {}
        self.role = ''
        self.mode = consts.HETERO

        self.model_param = BoostingTreeParam()

    def _init_model(self, boostingtree_param):
        self.tree_param = boostingtree_param.tree_param
        self.task_type = boostingtree_param.task_type
        self.objective_param = boostingtree_param.objective_param
        self.learning_rate = boostingtree_param.learning_rate
        self.num_trees = boostingtree_param.num_trees
        self.subsample_feature_rate = boostingtree_param.subsample_feature_rate
        self.n_iter_no_change = boostingtree_param.n_iter_no_change
        self.encrypt_param = boostingtree_param.encrypt_param
        self.tol = boostingtree_param.tol
        self.quantile_method = boostingtree_param.quantile_method
        self.bin_num = boostingtree_param.bin_num
        self.bin_gap = boostingtree_param.bin_gap
        self.bin_sample_num = boostingtree_param.bin_sample_num
        self.calculated_mode = boostingtree_param.encrypted_mode_calculator_param.mode
        self.re_encrypted_rate = boostingtree_param.encrypted_mode_calculator_param.re_encrypted_rate
        self.predict_param = boostingtree_param.predict_param
        self.cv_param = boostingtree_param.cv_param

    @staticmethod
    def data_format_transform(row):
        if type(row.features).__name__ != consts.SPARSE_VECTOR:
            feature_shape = row.features.shape[0]
            indices = []
            data = []

            for i in range(feature_shape):
                if np.abs(row.features[i]) < consts.FLOAT_ZERO:
                    continue

                indices.append(i)
                data.append(row.features[i])

            row.features = SparseVector(indices, data, feature_shape)

        return row

    def data_alignment(self, data_inst):
        abnormal_detection.empty_table_detection(data_inst)
        abnormal_detection.empty_feature_detection(data_inst)

        schema = data_inst.schema
        new_data_inst = data_inst.mapValues(lambda row: BoostingTree.data_format_transform(row))

        new_data_inst.schema = schema

        return new_data_inst

    def gen_feature_fid_mapping(self, schema):
        header = schema.get("header")
        for i in range(len(header)):
            self.feature_name_fid_mapping[header[i]] = i
    
    """
    def callback_meta(self, metric_namespace, metric_name, metric_meta):
        tracker = Tracking("abc", "123")
        tracker.set_metric_meta(metric_namespace,
                                metric_name,
                                metric_meta)

    def callback_metric(self, metric_namespace, metric_name, metric_data):
        tracker = Tracking("abc", "123")
        tracker.log_metric_data(metric_namespace,
                                metric_name,
                                metric_data)
    """

    def fit(self, data_inst):
        pass

    def predict(self, data_inst):
        pass

    def cross_validation(self, data_instances):
        if not self.need_run:
            return data_instances
        kflod_obj = KFold()
        cv_param = self._get_cv_param()
        kflod_obj.run(cv_param, data_instances, self)
        return data_instances

    def _get_cv_param(self):
        self.model_param.cv_param.role = self.role
        self.model_param.cv_param.mode = self.mode
        return self.model_param.cv_param

    def predict_proba(self, data_inst):
        pass

    def load_model(self):
        pass

    def save_data(self):
        return self.data_output

    def save_model(self):
        pass
