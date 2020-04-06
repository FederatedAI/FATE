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
# HeteroSecureBoostingHost 
# =============================================================================

from numpy import random

from arch.api.utils import log_utils
from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.feature.fate_element_type import NoneType
from federatedml.param.feature_binning_param import FeatureBinningParam
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import QuantileMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam
from federatedml.transfer_variable.transfer_class.hetero_secure_boost_transfer_variable import \
    HeteroSecureBoostingTreeTransferVariable
from federatedml.tree import BoostingTree
from federatedml.tree import HeteroDecisionTreeHost
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroSecureBoostingTreeHost(BoostingTree):
    def __init__(self):
        super(HeteroSecureBoostingTreeHost, self).__init__()

        self.transfer_variable = HeteroSecureBoostingTreeTransferVariable()
        # self.flowid = 0
        self.tree_dim = None
        self.feature_num = None
        self.trees_ = []
        self.tree_meta = None
        self.bin_split_points = None
        self.bin_sparse_points = None
        self.data_bin = None
        self.role = consts.HOST

    def convert_feature_to_bin(self, data_instance):
        LOGGER.info("convert feature to bins")
        param_obj = FeatureBinningParam(bin_num=self.bin_num)
        if self.use_missing:
            binning_obj = QuantileBinning(param_obj, abnormal_list=[NoneType()])
        else:
            binning_obj = QuantileBinning(param_obj)

        binning_obj.fit_split_points(data_instance)
        self.data_bin, self.bin_split_points, self.bin_sparse_points = binning_obj.convert_feature_to_bin(data_instance)

    def sample_valid_features(self):
        LOGGER.info("sample valid features")
        if self.feature_num is None:
            self.feature_num = self.bin_split_points.shape[0]

        choose_feature = random.choice(range(0, self.feature_num), \
                                       max(1, int(self.subsample_feature_rate * self.feature_num)), replace=False)

        valid_features = [False for i in range(self.feature_num)]
        for fid in choose_feature:
            valid_features[fid] = True
        return valid_features

    def generate_flowid(self, round_num, tree_num):
        LOGGER.info("generate flowid, flowid {}".format(self.flowid))
        return ".".join(map(str, [self.flowid, round_num, tree_num]))

    def sync_tree_dim(self):
        LOGGER.info("sync tree dim from guest")
        self.tree_dim = self.transfer_variable.tree_dim.get(idx=0)
        LOGGER.info("tree dim is %d" % (self.tree_dim))

    def sync_stop_flag(self, num_round):
        LOGGER.info("sync stop flag from guest, boosting round is {}".format(num_round))
        stop_flag = self.transfer_variable.stop_flag.get(idx=0,
                                                         suffix=(num_round,))
        return stop_flag

    def fit(self, data_inst, validate_data=None):

        LOGGER.info("begin to train secureboosting guest model")
        self.gen_feature_fid_mapping(data_inst.schema)
        LOGGER.debug("schema is {}".format(data_inst.schema))
        data_inst = self.data_alignment(data_inst)
        self.convert_feature_to_bin(data_inst)
        self.sync_tree_dim()

        self.validation_strategy = self.init_validation_strategy(data_inst, validate_data)

        for i in range(self.num_trees):
            # n_tree = []
            for tidx in range(self.tree_dim):
                tree_inst = HeteroDecisionTreeHost(self.tree_param)

                tree_inst.set_inputinfo(data_bin=self.data_bin, bin_split_points=self.bin_split_points,
                                        bin_sparse_points=self.bin_sparse_points)

                valid_features = self.sample_valid_features()
                tree_inst.set_flowid(self.generate_flowid(i, tidx))
                tree_inst.set_runtime_idx(self.component_properties.local_partyid)
                tree_inst.set_valid_features(valid_features)

                tree_inst.fit()
                tree_meta, tree_param = tree_inst.get_model()
                self.trees_.append(tree_param)
                if self.tree_meta is None:
                    self.tree_meta = tree_meta
                # n_tree.append(tree_inst.get_tree_model())

            # self.trees_.append(n_tree)

            if self.validation_strategy:
                LOGGER.debug('host running validation')
                self.validation_strategy.validate(self, i)
                if self.validation_strategy.need_stop():
                    LOGGER.debug('early stopping triggered')
                    break

            if self.n_iter_no_change is True:
                stop_flag = self.sync_stop_flag(i)
                if stop_flag:
                    break

        if self.validation_strategy and self.validation_strategy.has_saved_best_model():
            self.load_model(self.validation_strategy.cur_best_model)

        LOGGER.info("end to train secureboosting guest model")

    def predict(self, data_inst, predict_param=None):
        LOGGER.info("start predict")
        data_inst = self.data_alignment(data_inst)
        rounds = len(self.trees_) // self.tree_dim
        for i in range(rounds):
            # n_tree = self.trees_[i]
            for tidx in range(self.tree_dim):
                tree_inst = HeteroDecisionTreeHost(self.tree_param)
                tree_inst.load_model(self.tree_meta, self.trees_[i * self.tree_dim + tidx])
                # tree_inst.set_tree_model(self.trees_[i * self.tree_dim + tidx])
                tree_inst.set_flowid(self.generate_flowid(i, tidx))
                tree_inst.set_runtime_idx(self.component_properties.local_partyid)

                tree_inst.predict(data_inst)

        LOGGER.info("end predict")

    def get_model_meta(self):
        model_meta = BoostingTreeModelMeta()
        model_meta.tree_meta.CopyFrom(self.tree_meta)
        model_meta.num_trees = self.num_trees
        model_meta.quantile_meta.CopyFrom(QuantileMeta(bin_num=self.bin_num))

        meta_name = "HeteroSecureBoostingTreeHostMeta"

        return meta_name, model_meta

    def set_model_meta(self, model_meta):
        self.tree_meta = model_meta.tree_meta
        self.num_trees = model_meta.num_trees
        self.bin_num = model_meta.quantile_meta.bin_num

    def get_model_param(self):
        model_param = BoostingTreeModelParam()
        model_param.tree_num = len(list(self.trees_))
        model_param.tree_dim = self.tree_dim
        model_param.trees_.extend(self.trees_)
        LOGGER.debug("self.feature_name_fid_mapping is {}".format(self.feature_name_fid_mapping))
        model_param.feature_name_fid_mapping.update(self.feature_name_fid_mapping)

        param_name = "HeteroSecureBoostingTreeHostParam"

        return param_name, model_param

    def set_model_param(self, model_param):
        self.trees_ = list(model_param.trees_)
        self.tree_dim = model_param.tree_dim
        self.feature_name_fid_mapping.update(model_param.feature_name_fid_mapping)

    def export_model(self):
        if self.need_cv:
            return None

        meta_name, meta_protobuf = self.get_model_meta()
        param_name, param_protobuf = self.get_model_param()

        return {meta_name: meta_protobuf, param_name: param_protobuf}

    def load_model(self, model_dict):
        LOGGER.info("load model")
        model_param = None
        model_meta = None
        for _, value in model_dict["model"].items():
            for model in value:
                if model.endswith("Meta"):
                    model_meta = value[model]
                if model.endswith("Param"):
                    model_param = value[model]

        self.set_model_meta(model_meta)
        self.set_model_param(model_param)
