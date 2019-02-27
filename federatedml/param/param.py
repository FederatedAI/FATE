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
from federatedml.util import consts


class DataIOParam(object):
    def __init__(self, input_format="dense", delimitor=',', data_type='float64',
                 missing_fill=False, default_value=0, with_label=False, label_idx=0,
                 label_type='int', output_format='dense'):
        self.input_format = input_format
        self.delimitor = delimitor
        self.data_type = data_type
        self.missing_fill = missing_fill
        self.default_value = default_value
        self.with_label = with_label
        self.label_idx = label_idx
        self.label_type = label_type
        self.output_format = output_format


class EncryptParam(object):
    def __init__(self, method=consts.PAILLIER, key_length=1024):
        self.method = method
        self.key_length = key_length


class EvaluateParam(object):
    def __init__(self, metrics=None, classi_type="binary", pos_label=1, thresholds=None):
        self.metrics = metrics
        self.classi_type = classi_type
        self.pos_label = pos_label
        self.thresholds = thresholds


class PredictParam(object):
    def __init__(self, with_proba=True, threshold=0.5):
        self.with_proba = with_proba
        self.threshold = threshold


class WorkFlowParam(object):
    def __init__(self, method=None, train_input_table=None, train_input_namespace=None, model_table=None,
                 model_namespace=None, predict_input_table=None, predict_input_namespace=None,
                 predict_result_partition=1, predict_output_table=None, predict_output_namespace=None,
                 evaluation_output_table=None, evaluation_output_namespace=None,
                 data_input_table=None, data_input_namespace=None, intersect_data_output_table=None,
                 intersect_data_output_namespace=None, dataio_param=DataIOParam(), predict_param=PredictParam(),
                 evaluate_param=EvaluateParam(), do_cross_validation=False, work_mode=0,
                 n_splits=5):
        self.method = method
        self.train_input_table = train_input_table
        self.train_input_namespace = train_input_namespace
        self.model_table = model_table
        self.model_namespace = model_namespace
        self.predict_input_table = predict_input_table
        self.predict_input_namespace = predict_input_namespace
        self.predict_output_table = predict_output_table
        self.predict_output_namespace = predict_output_namespace
        self.predict_result_partition = predict_result_partition
        self.evaluation_output_table = evaluation_output_table
        self.evaluation_output_namespace = evaluation_output_namespace
        self.data_input_table = data_input_table
        self.data_input_namespace = data_input_namespace
        self.intersect_data_output_table = intersect_data_output_table
        self.intersect_data_output_namespace = intersect_data_output_namespace
        self.dataio_param = dataio_param
        self.do_cross_validation = do_cross_validation
        self.n_splits = n_splits
        self.work_mode = work_mode
        self.predict_param = predict_param
        self.evaluate_param = evaluate_param


class InitParam(object):
    def __init__(self, init_method='random_uniform', init_const=1, fit_intercept=False):
        self.init_method = init_method
        self.init_const = init_const
        self.fit_intercept = fit_intercept


class EncodeParam(object):
    def __init__(self, salt='', encode_method=None, base64=0):
        self.salt = salt
        self.encode_method = encode_method
        self.base64 = base64


class IntersectParam(object):
    def __init__(self, intersect_method=consts.RAW, random_bit=128, is_send_intersect_ids=True,
                 is_get_intersect_ids=True, join_role="guest", with_encode=False, encode_params=EncodeParam()):
        self.intersect_method = intersect_method
        self.random_bit = random_bit
        self.is_send_intersect_ids = is_send_intersect_ids
        self.is_get_intersect_ids = is_get_intersect_ids
        self.join_role = join_role
        self.with_encode = with_encode
        self.encode_params = encode_params


class LogisticParam(object):
    def __init__(self, penalty='L2',
                 eps=1e-5, alpha=1.0, optimizer='sgd', party_weight=1,
                 batch_size=-1, learning_rate=0.01, init_param=InitParam(),
                 max_iter=100, converge_func='diff',
                 encrypt_param=EncryptParam(), re_encrypt_batches=2,
                 model_path='lr_model', table_name='lr_table'):
        self.penalty = penalty
        self.eps = eps
        self.alpha = alpha
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.init_param = init_param
        self.max_iter = max_iter
        self.converge_func = converge_func
        self.encrypt_param = encrypt_param
        self.re_encrypt_batches = re_encrypt_batches
        self.model_path = model_path
        self.table_name = table_name
        self.party_weight = party_weight


class DecisionTreeParam(object):
    def __init__(self, criterion_method="xgboost", criterion_params=[0.1], max_depth=5,
                 min_sample_split=2, min_imputiry_split=1e-3, min_leaf_node=1, n_iter_no_change=True, tol=0.001):
        self.criterion_method = criterion_method
        self.criterion_params = criterion_params
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_impurity_split = min_imputiry_split
        self.min_leaf_node = min_leaf_node
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol


class BoostingTreeParam(object):
    def __init__(self, tree_param=DecisionTreeParam(), task_type="classification", loss_type="cross_entropy",
                 learning_rate=0.3, num_trees=5, subsample_feature_rate=0.8, n_iter_no_change=True,
                 tol=0.0001, encrypt_param=EncryptParam(), quantile_method="bin_by_sample_data",
                 bin_num=32, bin_gap=1e-3, bin_sample_num=10000):
        self.tree_param = tree_param
        self.task_type = task_type
        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.num_trees = num_trees
        self.subsample_feature_rate = subsample_feature_rate
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.encrypt_param = encrypt_param
        self.quantile_method = quantile_method
        self.bin_num = bin_num
        self.bin_gap = bin_gap
        self.bin_sample_num = bin_sample_num


class FTLModelParam(object):
    def __init__(self, max_iteration=10, batch_size=64, eps=1e-5,
                 alpha=100, lr_decay=0.001, l2_para=1, is_encrypt=True):
        self.max_iter = max_iteration
        self.batch_size = batch_size
        self.eps = eps
        self.alpha = alpha
        self.lr_decay = lr_decay
        self.l2_para = l2_para
        self.is_encrypt = is_encrypt


class FTLLocalModelParam(object):
    def __init__(self, encode_dim=5, learning_rate=0.001):
        self.encode_dim = encode_dim
        self.learning_rate = learning_rate


class FTLDataParam(object):
    def __init__(self, file_path=None, n_feature_guest=10, n_feature_host=23, overlap_ratio=0.1, guest_split_ratio=0.9,
                 num_samples=None, balanced=True, is_read_table=False):
        self.file_path = file_path
        self.n_feature_guest = n_feature_guest
        self.n_feature_host = n_feature_host
        self.overlap_ratio = overlap_ratio
        self.guest_split_ratio = guest_split_ratio
        self.num_samples = num_samples
        self.balanced = balanced
        self.is_read_table = is_read_table
