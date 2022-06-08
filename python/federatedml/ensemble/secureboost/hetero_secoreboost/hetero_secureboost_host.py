from operator import itemgetter
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.util.io_check import assert_io_num_rows_equal
from federatedml.util.anonymous_generator import generate_anonymous
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import FeatureImportanceInfo
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.feature_importance import FeatureImportance
from federatedml.ensemble.boosting import HeteroBoostingHost
from federatedml.param.boosting_param import HeteroSecureBoostParam, DecisionTreeParam
from federatedml.transfer_variable.transfer_class.hetero_secure_boosting_predict_transfer_variable import \
    HeteroSecureBoostTransferVariable
from federatedml.ensemble.secureboost.secureboost_util.tree_model_io import produce_hetero_tree_learner, \
    load_hetero_tree_learner
from federatedml.ensemble.secureboost.secureboost_util.boosting_tree_predict import sbt_host_predict, \
    mix_sbt_host_predict, EINI_host_predict
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import QuantileMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core import tree_plan as plan


class HeteroSecureBoostingTreeHost(HeteroBoostingHost):

    def __init__(self):
        super(HeteroSecureBoostingTreeHost, self).__init__()
        self.use_missing = False
        self.zero_as_missing = False
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

        self.boosting_strategy = consts.STD_TREE

        # fast sbt param
        self.tree_num_per_party = 1
        self.guest_depth = 0
        self.host_depth = 0
        self.init_tree_plan = False
        self.tree_plan = []
        self.feature_importances_ = {}

        # EINI predict param
        self.EINI_inference = False
        self.EINI_random_mask = False
        self.EINI_complexity_check = False

        self.multi_mode = consts.SINGLE_OUTPUT

        self.hetero_sbt_transfer_variable = HeteroSecureBoostTransferVariable()

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

        self.tree_num_per_party = param.tree_num_per_party
        self.boosting_strategy = param.boosting_strategy
        self.guest_depth = param.guest_depth
        self.host_depth = param.host_depth
        self.multi_mode = param.multi_mode
        self.EINI_inference = param.EINI_inference
        self.EINI_random_mask = param.EINI_random_mask
        self.EINI_complexity_check = param.EINI_complexity_check

        if self.use_missing:
            self.tree_param.use_missing = self.use_missing
            self.tree_param.zero_as_missing = self.zero_as_missing

    def get_tree_plan(self, idx):

        if not self.init_tree_plan:
            tree_plan = plan.create_tree_plan(self.boosting_strategy, k=self.tree_num_per_party,
                                              tree_num=self.boosting_round,
                                              host_list=self.component_properties.host_party_idlist,
                                              complete_secure=self.complete_secure)
            self.tree_plan += tree_plan
            self.init_tree_plan = True

        LOGGER.info('tree plan is {}'.format(self.tree_plan))
        return self.tree_plan[idx]

    def update_feature_importance(self, tree_feature_importance):
        for fid in tree_feature_importance:
            if fid not in self.feature_importances_:
                self.feature_importances_[fid] = tree_feature_importance[fid]
            else:
                self.feature_importances_[fid] += tree_feature_importance[fid]

    def load_feature_importance(self, feat_importance_param):
        param = list(feat_importance_param)
        rs_dict = {}
        for fp in param:
            key = fp.fid
            importance = FeatureImportance()
            importance.from_protobuf(fp)
            rs_dict[key] = importance

        self.feature_importances_ = rs_dict

    def preprocess(self):
        if self.multi_mode == consts.MULTI_OUTPUT:
            self.booster_dim = 1

    def postprocess(self):
        # generate anonymous
        new_feat_importance = {}
        sitename = 'host:' + str(self.component_properties.local_partyid)
        for key in self.feature_importances_:
            new_feat_importance[(sitename, key)] = self.feature_importances_[key]
        self.hetero_sbt_transfer_variable.host_feature_importance.remote(new_feat_importance)

    def fit_a_learner(self, epoch_idx: int, booster_dim: int):

        flow_id = self.generate_flowid(epoch_idx, booster_dim)
        complete_secure = True if (epoch_idx == 0 and self.complete_secure) else False
        fast_sbt = (self.boosting_strategy == consts.MIX_TREE or self.boosting_strategy == consts.LAYERED_TREE)

        tree_type, target_host_id = None, None
        if fast_sbt:
            tree_type, target_host_id = self.get_tree_plan(epoch_idx)

        tree = produce_hetero_tree_learner(role=self.role, tree_param=self.tree_param, flow_id=flow_id,
                                           data_bin=self.data_bin, bin_split_points=self.bin_split_points,
                                           bin_sparse_points=self.bin_sparse_points, task_type=self.task_type,
                                           valid_features=self.sample_valid_features(),
                                           host_party_list=self.component_properties.host_party_idlist,
                                           runtime_idx=self.component_properties.local_partyid,
                                           cipher_compress=self.cipher_compressing,
                                           complete_secure=complete_secure,
                                           fast_sbt=fast_sbt, tree_type=tree_type, target_host_id=target_host_id,
                                           guest_depth=self.guest_depth, host_depth=self.host_depth,
                                           mo_tree=(self.multi_mode == consts.MULTI_OUTPUT), bin_num=self.bin_num
                                           )
        tree.fit()
        self.update_feature_importance(tree.get_feature_importance())
        return tree

    def load_learner(self, model_meta, model_param, epoch_idx, booster_idx):

        flow_id = self.generate_flowid(epoch_idx, booster_idx)
        runtime_idx = self.component_properties.local_partyid
        fast_sbt = (self.boosting_strategy == consts.MIX_TREE or self.boosting_strategy == consts.LAYERED_TREE)
        tree_type, target_host_id = None, None

        if fast_sbt:
            tree_type, target_host_id = self.get_tree_plan(epoch_idx)

        tree = load_hetero_tree_learner(self.role, self.tree_param, model_meta, model_param, flow_id,
                                        runtime_idx,
                                        fast_sbt=fast_sbt, tree_type=tree_type, target_host_id=target_host_id)

        return tree

    def generate_summary(self) -> dict:

        summary = {'best_iteration': self.callback_variables.best_iteration, 'is_converged': self.is_converged}
        LOGGER.debug('summary is {}'.format(summary))

        return summary

    @assert_io_num_rows_equal
    def predict(self, data_inst):

        LOGGER.info('running prediction')

        processed_data = self.data_and_header_alignment(data_inst)

        predict_start_round = self.sync_predict_start_round()

        rounds = len(self.boosting_model_list) // self.booster_dim
        trees = []
        for idx in range(predict_start_round, rounds):
            for booster_idx in range(self.booster_dim):
                tree = self.load_learner(self.booster_meta,
                                         self.boosting_model_list[idx * self.booster_dim + booster_idx],
                                         idx, booster_idx)
                trees.append(tree)

        if len(trees) == 0:
            LOGGER.info('no tree for predicting, prediction done')
            return

        if self.boosting_strategy == consts.MIX_TREE:
            mix_sbt_host_predict(processed_data, self.hetero_sbt_transfer_variable, trees)
        else:
            if self.EINI_inference and not self.on_training:
                sitename = self.role + ':' + str(self.component_properties.local_partyid)
                EINI_host_predict(processed_data, trees, sitename, self.component_properties.local_partyid,
                                  self.component_properties.host_party_idlist, self.booster_dim,
                                  self.hetero_sbt_transfer_variable, self.EINI_complexity_check, self.EINI_random_mask,)
            else:
                sbt_host_predict(processed_data, self.hetero_sbt_transfer_variable, trees)

    def get_model_meta(self):
        model_meta = BoostingTreeModelMeta()
        model_meta.tree_meta.CopyFrom(self.booster_meta)
        model_meta.num_trees = self.boosting_round
        model_meta.quantile_meta.CopyFrom(QuantileMeta(bin_num=self.bin_num))
        model_meta.boosting_strategy = self.boosting_strategy
        model_meta.module = "HeteroSecureBoost"
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
            anonymous_name_mapping[generate_anonymous(fid, role=consts.HOST, party_id=party_id, )] = name

        model_param.anonymous_name_mapping.update(anonymous_name_mapping)
        model_param.feature_name_fid_mapping.update(self.feature_name_fid_mapping)
        if self.boosting_strategy == consts.STD_TREE:
            model_param.model_name = consts.HETERO_SBT
        elif self.boosting_strategy == consts.LAYERED_TREE:
            model_param.model_name = consts.HETERO_FAST_SBT_LAYERED
        elif self.boosting_strategy == consts.MIX_TREE:
            model_param.model_name = consts.HETERO_FAST_SBT_MIX
        model_param.best_iteration = self.callback_variables.best_iteration
        model_param.tree_plan.extend(plan.encode_plan(self.tree_plan))

        feature_importances = list(self.feature_importances_.items())
        feature_importances = sorted(feature_importances, key=itemgetter(1), reverse=True)
        feature_importance_param = []
        LOGGER.debug('host feat importance is {}'.format(feature_importances))
        for fid, importance in feature_importances:
            feature_importance_param.append(FeatureImportanceInfo(sitename=consts.HOST_LOCAL,  # host local feat
                                                                  fid=fid,
                                                                  importance=importance.importance,
                                                                  fullname=self.feature_name_fid_mapping[fid],
                                                                  main=importance.main_type
                                                                  ))
        model_param.feature_importances.extend(feature_importance_param)

        param_name = "HeteroSecureBoostingTreeHostParam"

        return param_name, model_param

    def set_model_meta(self, model_meta):
        if not self.is_warm_start:
            self.boosting_round = model_meta.num_trees
        self.booster_meta = model_meta.tree_meta
        self.bin_num = model_meta.quantile_meta.bin_num
        self.boosting_strategy = model_meta.boosting_strategy

    def set_model_param(self, model_param):
        self.boosting_model_list = list(model_param.trees_)
        self.booster_dim = model_param.tree_dim
        self.feature_name_fid_mapping.update(model_param.feature_name_fid_mapping)
        self.tree_plan = plan.decode_plan(model_param.tree_plan)
        self.load_feature_importance(model_param.feature_importances)

    # implement abstract function
    def check_label(self, *args):
        pass
