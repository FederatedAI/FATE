from arch.api.utils import log_utils
from federatedml.util import consts
from typing import List
import functools
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import QuantileMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam
from federatedml.ensemble.boosting.boosting_core import HeteroBoostingHost
from federatedml.param.boosting_param import HeteroSecureBoostParam
from federatedml.ensemble.basic_algorithms import HeteroDecisionTreeHost
from federatedml.transfer_variable.transfer_class.hetero_secure_boosting_predict_transfer_variable import \
    HeteroSecureBoostTransferVariable
from federatedml.util.io_check import assert_io_num_rows_equal
from federatedml.util.fate_operator import generate_anonymous


LOGGER = log_utils.getLogger()


class HeteroSecureBoostHost(HeteroBoostingHost):

    def __init__(self):
        super(HeteroSecureBoostHost, self).__init__()
        self.tree_param = None  # decision tree param
        self.use_missing = False
        self.zero_as_missing = False
        self.cur_epoch_idx = -1
        self.grad_and_hess = None
        self.model_param = HeteroSecureBoostParam()
        self.complete_secure = False

        self.predict_transfer_inst = HeteroSecureBoostTransferVariable()

    def _init_model(self, param: HeteroSecureBoostParam):
        super(HeteroSecureBoostHost, self)._init_model(param)
        self.tree_param = param.tree_param
        self.use_missing = param.use_missing
        self.zero_as_missing = param.zero_as_missing
        self.complete_secure = param.complete_secure

        if self.use_missing:
            self.tree_param.use_missing = self.use_missing
            self.tree_param.zero_as_missing = self.zero_as_missing

    def fit_a_booster(self, epoch_idx: int, booster_dim: int):

        tree = HeteroDecisionTreeHost(tree_param=self.tree_param)
        tree.set_input_data(data_bin=self.data_bin, bin_split_points=self.bin_split_points, bin_sparse_points=
                            self.bin_sparse_points)
        tree.set_valid_features(self.sample_valid_features())
        tree.set_flowid(self.generate_flowid(epoch_idx, booster_dim))
        tree.set_runtime_idx(self.component_properties.local_partyid)

        if self.complete_secure and epoch_idx == 0:
            tree.set_as_complete_secure_tree()

        tree.fit()

        return tree

    def load_booster(self, model_meta, model_param, epoch_idx, booster_idx):
        tree = HeteroDecisionTreeHost(self.tree_param)
        tree.load_model(model_meta, model_param)
        tree.set_flowid(self.generate_flowid(epoch_idx, booster_idx))
        tree.set_runtime_idx(self.component_properties.local_partyid)
        return tree

    def generate_summary(self) -> dict:

        summary = {'best_iteration': self.validation_strategy.best_iteration, 'is_converged': self.is_converged}

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

        for t_idx, tree in enumerate(trees):

            cur_node_idx = leaf_pos['node_pos'][t_idx]
            # idx is set as -1 when a sample reaches leaf
            if cur_node_idx == -1:
                continue
            nid, _ = HeteroSecureBoostHost.traverse_a_tree(tree, sample, cur_node_idx)
            leaf_pos['node_pos'][t_idx] = nid

        return leaf_pos

    def boosting_fast_predict(self, data_inst, trees: List[HeteroDecisionTreeHost]):

        comm_round = 0

        traverse_func = functools.partial(self.traverse_trees, trees=trees)

        while True:

            LOGGER.debug('cur predict round is {}'.format(comm_round))

            stop_flag = self.predict_transfer_inst.predict_stop_flag.get(idx=0, suffix=(comm_round, ))
            if stop_flag:
                break

            guest_node_pos = self.predict_transfer_inst.guest_predict_data.get(idx=0, suffix=(comm_round, ))
            host_node_pos = guest_node_pos.join(data_inst, traverse_func)
            self.predict_transfer_inst.host_predict_data.remote(host_node_pos, idx=-1, suffix=(comm_round, ))

            comm_round += 1

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
        for name, fid in self.feature_name_fid_mapping.items():
            anonymous_name_mapping = {generate_anonymous(fid, role=consts.HOST, party_id=party_id,):
                                                name}

        model_param.anonymous_name_mapping.update(anonymous_name_mapping)
        model_param.feature_name_fid_mapping.update(self.feature_name_fid_mapping)
        model_param.model_name = consts.HETERO_SBT
        model_param.best_iteration = -1 if self.validation_strategy is None else self.validation_strategy.best_iteration

        param_name = "HeteroSecureBoostingTreeHostParam"

        return param_name, model_param

    def set_model_meta(self, model_meta):
        self.booster_meta = model_meta.tree_meta
        self.boosting_round = model_meta.num_trees
        self.bin_num = model_meta.quantile_meta.bin_num

    def set_model_param(self, model_param):
        self.boosting_model_list = list(model_param.trees_)
        self.booster_dim = model_param.tree_dim
        self.feature_name_fid_mapping.update(model_param.feature_name_fid_mapping)