from typing import List
import functools
import copy
import random
import numpy as np
from scipy import sparse as sp
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import QuantileMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam
from federatedml.ensemble.boosting.boosting_core import HeteroBoostingHost
from federatedml.param.boosting_param import HeteroSecureBoostParam, DecisionTreeParam
from federatedml.ensemble.basic_algorithms import HeteroDecisionTreeHost
from federatedml.transfer_variable.transfer_class.hetero_secure_boosting_predict_transfer_variable import \
    HeteroSecureBoostTransferVariable
from federatedml.util.io_check import assert_io_num_rows_equal
from federatedml.util.anonymous_generator import generate_anonymous
from federatedml.feature.fate_element_type import NoneType


class HeteroSecureBoostingTreeHost(HeteroBoostingHost):

    def __init__(self):
        super(HeteroSecureBoostingTreeHost, self).__init__()
        self.use_missing = False
        self.zero_as_missing = False
        self.cur_epoch_idx = -1
        self.grad_and_hess = None
        self.tree_param = DecisionTreeParam()  # decision tree param
        self.model_param = HeteroSecureBoostParam()
        self.complete_secure = False
        self.model_name = 'HeteroSecureBoost'
        self.enable_goss = False
        self.cipher_compressing = False
        self.max_sample_weight = None
        self.round_decimal = None
        self.new_ver = True
        self.feature_importances_ = {}

        # for fast hist
        self.sparse_opt_para = False
        self.run_sparse_opt = False
        self.has_transformed_data = False
        self.data_bin_dense = None
        self.hetero_sbt_transfer_variable = HeteroSecureBoostTransferVariable()

        # EINI predict param
        self.EINI_inference = False
        self.EINI_random_mask = False
        self.EINI_complexity_check = False

    def _init_model(self, param: HeteroSecureBoostParam):

        super(HeteroSecureBoostingTreeHost, self)._init_model(param)
        self.tree_param = param.tree_param
        self.use_missing = param.use_missing
        self.enable_goss = param.run_goss
        self.zero_as_missing = param.zero_as_missing
        self.complete_secure = param.complete_secure
        self.sparse_opt_para = param.sparse_optimization
        self.cipher_compressing = param.cipher_compress
        self.new_ver = param.new_ver
        self.EINI_inference = param.EINI_inference
        self.EINI_random_mask = param.EINI_random_mask
        self.EINI_complexity_check = param.EINI_complexity_check

        if self.use_missing:
            self.tree_param.use_missing = self.use_missing
            self.tree_param.zero_as_missing = self.zero_as_missing

    @staticmethod
    def sparse_to_array(data, feature_sparse_point_array, use_missing, zero_as_missing):
        new_data = copy.deepcopy(data)
        new_feature_sparse_point_array = copy.deepcopy(feature_sparse_point_array)
        for k, v in data.features.get_all_data():
            if v == NoneType():
                value = -1
            else:
                value = v
            new_feature_sparse_point_array[k] = value

        # as most sparse point is bin-0
        # when mark it as a missing value (-1), offset it to make it sparse
        if not use_missing or (use_missing and not zero_as_missing):
            offset = 0
        else:
            offset = 1
        new_data.features = sp.csc_matrix(np.array(new_feature_sparse_point_array) + offset)
        return new_data

    def update_feature_importance(self, tree_feature_importance):
        for fid in tree_feature_importance:
            if fid not in self.feature_importances_:
                self.feature_importances_[fid] = tree_feature_importance[fid]
            else:
                self.feature_importances_[fid] += tree_feature_importance[fid]
        LOGGER.debug('cur feature importance {}'.format(self.feature_importances_))

    def sync_feature_importance(self):
        # generate anonymous
        new_feat_importance = {}
        sitename = 'host:' + str(self.component_properties.local_partyid)
        for key in self.feature_importances_:
            new_feat_importance[(sitename, key)] = self.feature_importances_[key]
        self.hetero_sbt_transfer_variable.host_feature_importance.remote(new_feat_importance)

    def fit_a_booster(self, epoch_idx: int, booster_dim: int):

        tree = HeteroDecisionTreeHost(tree_param=self.tree_param)
        tree.init(flowid=self.generate_flowid(epoch_idx, booster_dim),
                  valid_features=self.sample_valid_features(),
                  data_bin=self.data_bin, bin_split_points=self.bin_split_points,
                  bin_sparse_points=self.bin_sparse_points,
                  runtime_idx=self.component_properties.local_partyid,
                  goss_subsample=self.enable_goss,
                  bin_num=self.bin_num,
                  complete_secure=True if (self.complete_secure and epoch_idx == 0) else False,
                  cipher_compressing=self.cipher_compressing,
                  new_ver=self.new_ver
                  )
        tree.fit()
        self.update_feature_importance(tree.get_feature_importance())
        self.sync_feature_importance()

        return tree

    def load_booster(self, model_meta, model_param, epoch_idx, booster_idx):
        tree = HeteroDecisionTreeHost(self.tree_param)
        tree.load_model(model_meta, model_param)
        tree.set_flowid(self.generate_flowid(epoch_idx, booster_idx))
        tree.set_runtime_idx(self.component_properties.local_partyid)
        return tree

    def generate_summary(self) -> dict:

        summary = {'best_iteration': self.callback_variables.best_iteration, 'is_converged': self.is_converged}
        LOGGER.debug('summary is {}'.format(summary))

        return summary

    @staticmethod
    def traverse_a_tree(tree: HeteroDecisionTreeHost, sample, cur_node_idx):

        nid, _ = tree.traverse_tree(predict_state=(cur_node_idx, -1), data_inst=sample,
                                    decoder=tree.decode, split_maskdict=tree.split_maskdict,
                                    missing_dir_maskdict=tree.missing_dir_maskdict, sitename=tree.sitename,
                                    tree_=tree.tree_node, zero_as_missing=tree.zero_as_missing,
                                    use_missing=tree.use_missing)

        return nid, _

    @staticmethod
    def traverse_trees(leaf_pos, sample, trees: List[HeteroDecisionTreeHost]):

        new_leaf_pos = {'node_pos': leaf_pos['node_pos'], 'reach_leaf_node': leaf_pos['reach_leaf_node'] + False}
        for t_idx, tree in enumerate(trees):

            cur_node_idx = new_leaf_pos['node_pos'][t_idx]
            # idx is set as -1 when a sample reaches leaf
            if cur_node_idx == -1:
                continue
            nid, _ = HeteroSecureBoostingTreeHost.traverse_a_tree(tree, sample, cur_node_idx)
            new_leaf_pos['node_pos'][t_idx] = nid

        return new_leaf_pos

    def boosting_fast_predict(self, data_inst, trees: List[HeteroDecisionTreeHost]):

        comm_round = 0

        traverse_func = functools.partial(self.traverse_trees, trees=trees)

        while True:

            LOGGER.debug('cur predict round is {}'.format(comm_round))

            stop_flag = self.hetero_sbt_transfer_variable.predict_stop_flag.get(idx=0, suffix=(comm_round,))
            if stop_flag:
                break

            guest_node_pos = self.hetero_sbt_transfer_variable.guest_predict_data.get(idx=0, suffix=(comm_round,))
            host_node_pos = guest_node_pos.join(data_inst, traverse_func)
            if guest_node_pos.count() != host_node_pos.count():
                raise ValueError('sample count mismatch: guest table {}, host table {}'.format(guest_node_pos.count(),
                                                                                               host_node_pos.count()))
            self.hetero_sbt_transfer_variable.host_predict_data.remote(host_node_pos, idx=-1, suffix=(comm_round,))

            comm_round += 1

    @staticmethod
    def go_to_children_branches(data_inst, tree_node, tree, sitename: str, candidate_list: List):
        if tree_node.is_leaf:
            candidate_list.append(tree_node)
        else:
            tree_node_list = tree.tree_node
            if tree_node.sitename != sitename:
                HeteroSecureBoostingTreeHost.go_to_children_branches(data_inst, tree_node_list[tree_node.left_nodeid],
                                                                     tree, sitename, candidate_list)
                HeteroSecureBoostingTreeHost.go_to_children_branches(data_inst, tree_node_list[tree_node.right_nodeid],
                                                                     tree, sitename, candidate_list)
            else:
                next_layer_node_id = tree.go_next_layer(tree_node, data_inst, use_missing=tree.use_missing,
                                                        zero_as_missing=tree.zero_as_missing, decoder=tree.decode,
                                                        split_maskdict=tree.split_maskdict,
                                                        missing_dir_maskdict=tree.missing_dir_maskdict,
                                                        bin_sparse_point=None
                                                        )
                HeteroSecureBoostingTreeHost.go_to_children_branches(data_inst, tree_node_list[next_layer_node_id],
                                                                     tree, sitename, candidate_list)

    @staticmethod
    def generate_leaf_candidates_host(data_inst, sitename, trees, node_pos_map_list):
        candidate_nodes_of_all_tree = []

        for tree, node_pos_map in zip(trees, node_pos_map_list):

            result_vec = [0 for i in range(len(node_pos_map))]
            candidate_list = []
            HeteroSecureBoostingTreeHost.go_to_children_branches(data_inst, tree.tree_node[0], tree, sitename,
                                                                 candidate_list)
            for node in candidate_list:
                result_vec[node_pos_map[node.id]] = 1  # create 0-1 vector
            candidate_nodes_of_all_tree.extend(result_vec)

        return np.array(candidate_nodes_of_all_tree)

    @staticmethod
    def generate_leaf_idx_dimension_map(trees, booster_dim):
        cur_dim = 0
        leaf_dim_map = {}
        leaf_idx = 0
        for tree in trees:
            for node in tree.tree_node:
                if node.is_leaf:
                    leaf_dim_map[leaf_idx] = cur_dim
                    leaf_idx += 1
            cur_dim += 1
            if cur_dim == booster_dim:
                cur_dim = 0
        return leaf_dim_map

    @staticmethod
    def merge_position_vec(host_vec, guest_encrypt_vec, booster_dim=1, leaf_idx_dim_map=None, random_mask=None):

        leaf_idx = -1
        rs = [0 for i in range(booster_dim)]
        for en_num, vec_value in zip(guest_encrypt_vec, host_vec):
            leaf_idx += 1
            if vec_value == 0:
                continue
            else:
                dim = leaf_idx_dim_map[leaf_idx]
                rs[dim] += en_num

        if random_mask:
            for i in range(len(rs)):
                rs[i] = rs[i] * random_mask  # a pos random mask btw 1 and 2

        return rs

    @staticmethod
    def position_vec_element_wise_mul(guest_encrypt_vec, host_vec):
        new_vec = []
        for en_num, vec_value in zip(guest_encrypt_vec, host_vec):
            new_vec.append(en_num * vec_value)
        return new_vec

    @staticmethod
    def get_leaf_idx_map(trees):

        id_pos_map_list = []
        for tree in trees:
            array_idx = 0
            id_pos_map = {}
            for node in tree.tree_node:
                if node.is_leaf:
                    id_pos_map[node.id] = array_idx
                    array_idx += 1
            id_pos_map_list.append(id_pos_map)

        return id_pos_map_list

    def count_complexity_helper(self, node, node_list, host_sitename, meet_host_node):

        if node.is_leaf:
            return 1 if meet_host_node else 0
        if node.sitename == host_sitename:
            meet_host_node = True

        return self.count_complexity_helper(node_list[node.left_nodeid], node_list, host_sitename, meet_host_node) + \
               self.count_complexity_helper(node_list[node.right_nodeid], node_list, host_sitename, meet_host_node)

    def count_complexity(self, trees):

        tree_valid_leaves_num = []
        sitename = self.role + ":" + str(self.component_properties.local_partyid)
        for tree in trees:
            valid_leaf_num = self.count_complexity_helper(tree.tree_node[0], tree.tree_node, sitename, False)
            if valid_leaf_num != 0:
                tree_valid_leaves_num.append(valid_leaf_num)

        complexity = 1
        for num in tree_valid_leaves_num:
            complexity *= num

        return complexity

    def EINI_host_predict(self, data_inst, trees: List[HeteroDecisionTreeHost], sitename, self_party_id, party_list,
                          random_mask=False):

        if self.EINI_complexity_check:
            complexity = self.count_complexity(trees)
            LOGGER.debug('checking EINI complexity: {}'.format(complexity))
            if complexity < consts.EINI_TREE_COMPLEXITY:
                raise ValueError('tree complexity: {}, is lower than safe '
                                 'threshold, inference is not allowed.'.format(complexity))
        id_pos_map_list = self.get_leaf_idx_map(trees)
        map_func = functools.partial(self.generate_leaf_candidates_host, sitename=sitename, trees=trees,
                                     node_pos_map_list=id_pos_map_list)
        position_vec = data_inst.mapValues(map_func)

        booster_dim = self.booster_dim
        random_mask = random.SystemRandom().random() + 1 if random_mask else 0  # generate a random mask btw 1 and 2

        self_idx = party_list.index(self_party_id)
        if len(party_list) == 1:
            guest_position_vec = self.hetero_sbt_transfer_variable.guest_predict_data.get(idx=0, suffix='position_vec')
            leaf_idx_dim_map = self.generate_leaf_idx_dimension_map(trees, booster_dim)
            merge_func = functools.partial(self.merge_position_vec, booster_dim=booster_dim,
                                           leaf_idx_dim_map=leaf_idx_dim_map, random_mask=random_mask)
            result_table = position_vec.join(guest_position_vec, merge_func)
            self.hetero_sbt_transfer_variable.host_predict_data.remote(result_table, suffix='merge_result')
        else:
            # multi host case
            # if is first host party, get encrypt vec from guest, else from previous host party
            if self_party_id == party_list[0]:
                guest_position_vec = self.hetero_sbt_transfer_variable.guest_predict_data.get(idx=0,
                                                                                              suffix='position_vec')
            else:
                guest_position_vec = self.hetero_sbt_transfer_variable.inter_host_data.get(idx=self_idx - 1,
                                                                                           suffix='position_vec')

            if self_party_id == party_list[-1]:
                leaf_idx_dim_map = self.generate_leaf_idx_dimension_map(trees, booster_dim)
                func = functools.partial(self.merge_position_vec, booster_dim=booster_dim,
                                         leaf_idx_dim_map=leaf_idx_dim_map, random_mask=random_mask)
                result_table = position_vec.join(guest_position_vec, func)
                self.hetero_sbt_transfer_variable.host_predict_data.remote(result_table, suffix='merge_result')
            else:
                result_table = position_vec.join(guest_position_vec, self.position_vec_element_wise_mul)
                self.hetero_sbt_transfer_variable.inter_host_data.remote(result_table, idx=self_idx + 1,
                                                                         suffix='position_vec',
                                                                         role=consts.HOST)

    @assert_io_num_rows_equal
    def predict(self, data_inst):

        LOGGER.info('running prediction')

        processed_data = self.data_and_header_alignment(data_inst)

        predict_start_round = self.sync_predict_start_round()

        rounds = len(self.boosting_model_list) // self.booster_dim
        trees = []
        for idx in range(predict_start_round, rounds):
            for booster_idx in range(self.booster_dim):
                tree = self.load_booster(self.booster_meta,
                                         self.boosting_model_list[idx * self.booster_dim + booster_idx],
                                         idx, booster_idx)
                trees.append(tree)

        if len(trees) == 0:
            LOGGER.info('no tree for predicting, prediction done')
            return

        if self.EINI_inference and not self.on_training:  # EINI is designed for inference stage
            sitename = self.role + ':' + str(self.component_properties.local_partyid)
            self.EINI_host_predict(processed_data, trees, sitename, self.component_properties.local_partyid,
                                   self.component_properties.host_party_idlist, self.EINI_random_mask)
        else:
            self.boosting_fast_predict(processed_data, trees=trees)

    def get_model_meta(self):
        model_meta = BoostingTreeModelMeta()
        model_meta.tree_meta.CopyFrom(self.booster_meta)
        model_meta.num_trees = self.boosting_round
        model_meta.quantile_meta.CopyFrom(QuantileMeta(bin_num=self.bin_num))
        meta_name = "HeteroSecureBoostingTreeHostMeta"
        return meta_name, model_meta

    def get_model_param(self):

        model_param = BoostingTreeModelParam()
        model_param.tree_num = len(self.boosting_model_list)
        model_param.tree_dim = self.booster_dim
        model_param.trees_.extend(self.boosting_model_list)

        anonymous_name_mapping = {}
        party_id = self.component_properties.local_partyid
        for fid, name in self.feature_name_fid_mapping.items():
            anonymous_name_mapping[generate_anonymous(fid, role=consts.HOST, party_id=party_id,)] = name

        model_param.anonymous_name_mapping.update(anonymous_name_mapping)
        model_param.feature_name_fid_mapping.update(self.feature_name_fid_mapping)
        model_param.model_name = consts.HETERO_SBT

        model_param.anonymous_name_mapping.update(anonymous_name_mapping)
        model_param.feature_name_fid_mapping.update(self.feature_name_fid_mapping)
        model_param.model_name = consts.HETERO_SBT
        model_param.best_iteration = self.callback_variables.best_iteration

        param_name = "HeteroSecureBoostingTreeHostParam"

        return param_name, model_param

    def set_model_meta(self, model_meta):
        if not self.is_warm_start:
            self.boosting_round = model_meta.num_trees
        self.booster_meta = model_meta.tree_meta
        self.bin_num = model_meta.quantile_meta.bin_num

    def set_model_param(self, model_param):
        self.boosting_model_list = list(model_param.trees_)
        self.booster_dim = model_param.tree_dim
        self.feature_name_fid_mapping.update(model_param.feature_name_fid_mapping)