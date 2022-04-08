import numpy as np
from operator import itemgetter
from federatedml.util import consts
from federatedml.util import LOGGER
from federatedml.ensemble.boosting import HeteroBoostingGuest
from federatedml.param.boosting_param import HeteroSecureBoostParam, DecisionTreeParam
from federatedml.util.io_check import assert_io_num_rows_equal
from federatedml.util.anonymous_generator import generate_anonymous
from federatedml.statistic.data_overview import with_weight, get_max_sample_weight
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core.feature_importance import FeatureImportance
from federatedml.transfer_variable.transfer_class.hetero_secure_boosting_predict_transfer_variable import \
    HeteroSecureBoostTransferVariable
from federatedml.ensemble.basic_algorithms.decision_tree.tree_core import tree_plan as plan
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import BoostingTreeModelMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import ObjectiveMeta
from federatedml.protobuf.generated.boosting_tree_model_meta_pb2 import QuantileMeta
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import BoostingTreeModelParam
from federatedml.protobuf.generated.boosting_tree_model_param_pb2 import FeatureImportanceInfo
from federatedml.ensemble.secureboost.secureboost_util.tree_model_io import load_hetero_tree_learner, \
    produce_hetero_tree_learner
from federatedml.ensemble.secureboost.secureboost_util.boosting_tree_predict import sbt_guest_predict, \
    mix_sbt_guest_predict, EINI_guest_predict
from federatedml.ensemble.secureboost.secureboost_util.subsample import goss_sampling


class HeteroSecureBoostingTreeGuest(HeteroBoostingGuest):

    def __init__(self):
        super(HeteroSecureBoostingTreeGuest, self).__init__()

        self.tree_param = DecisionTreeParam()  # decision tree param
        self.use_missing = False
        self.zero_as_missing = False
        self.cur_epoch_idx = -1
        self.grad_and_hess = None
        self.feature_importances_ = {}
        self.model_param = HeteroSecureBoostParam()
        self.complete_secure = False
        self.data_alignment_map = {}
        self.hetero_sbt_transfer_variable = HeteroSecureBoostTransferVariable()
        self.model_name = 'HeteroSecureBoost'
        self.max_sample_weight = 1
        self.max_sample_weight_computed = False
        self.re_compute_goss_sample_weight = False
        self.cipher_compressing = False

        self.enable_goss = False  # GOSS
        self.top_rate = None
        self.other_rate = None
        self.new_ver = True

        self.boosting_strategy = consts.STD_TREE  # default work mode is std

        # fast sbt param
        self.tree_num_per_party = 1
        self.guest_depth = 0
        self.host_depth = 0
        self.init_tree_plan = False
        self.tree_plan = []

        # multi-classification mode
        self.multi_mode = consts.SINGLE_OUTPUT

        # EINI predict param
        self.EINI_inference = False
        self.EINI_random_mask = False

    def _init_model(self, param: HeteroSecureBoostParam):

        super(HeteroSecureBoostingTreeGuest, self)._init_model(param)
        self.tree_param = param.tree_param
        self.use_missing = param.use_missing
        self.zero_as_missing = param.zero_as_missing
        self.complete_secure = param.complete_secure
        self.enable_goss = param.run_goss
        self.top_rate = param.top_rate
        self.other_rate = param.other_rate
        self.cipher_compressing = param.cipher_compress
        self.new_ver = param.new_ver
        self.EINI_inference = param.EINI_inference
        self.EINI_random_mask = param.EINI_random_mask

        # fast sbt param
        self.tree_num_per_party = param.tree_num_per_party
        self.boosting_strategy = param.boosting_strategy
        self.guest_depth = param.guest_depth
        self.host_depth = param.host_depth

        if self.use_missing:
            self.tree_param.use_missing = self.use_missing
            self.tree_param.zero_as_missing = self.zero_as_missing

        self.multi_mode = param.multi_mode

    def process_sample_weights(self, grad_and_hess, data_with_sample_weight=None):

        # add sample weights to gradient and hessian
        if data_with_sample_weight is not None:
            if with_weight(data_with_sample_weight):
                LOGGER.info('weighted sample detected, multiply g/h by weights')
                grad_and_hess = grad_and_hess.join(data_with_sample_weight,
                                                   lambda v1, v2: (v1[0] * v2.weight, v1[1] * v2.weight))
                if not self.max_sample_weight_computed:
                    self.max_sample_weight = get_max_sample_weight(data_with_sample_weight)
                    LOGGER.info('max sample weight is {}'.format(self.max_sample_weight))
                    self.max_sample_weight_computed = True

        return grad_and_hess

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

    def check_host_number(self, tree_type):
        host_num = len(self.component_properties.host_party_idlist)
        LOGGER.info('host number is {}'.format(host_num))
        if tree_type == plan.tree_type_dict['layered_tree']:
            assert host_num == 1, 'only 1 host party is allowed in layered mode'

    def compute_grad_and_hess(self, y_hat, y, data_with_sample_weight=None):

        LOGGER.info("compute grad and hess")
        loss_method = self.loss
        if self.task_type == consts.CLASSIFICATION:
            grad_and_hess = y.join(y_hat, lambda y, f_val:
                                   (loss_method.compute_grad(y, loss_method.predict(f_val)),
                                    loss_method.compute_hess(y, loss_method.predict(f_val))))
        else:
            grad_and_hess = y.join(y_hat, lambda y, f_val:
                                   (loss_method.compute_grad(y, f_val),
                                    loss_method.compute_hess(y, f_val)))

        grad_and_hess = self.process_sample_weights(grad_and_hess, data_with_sample_weight)

        return grad_and_hess

    @staticmethod
    def get_grad_and_hess(g_h, dim=0):
        LOGGER.info("get grad and hess of tree {}".format(dim))
        grad_and_hess_subtree = g_h.mapValues(
            lambda grad_and_hess: (grad_and_hess[0][dim], grad_and_hess[1][dim]))
        return grad_and_hess_subtree

    def update_feature_importance(self, tree_feature_importance):
        for fid in tree_feature_importance:
            if fid not in self.feature_importances_:
                self.feature_importances_[fid] = tree_feature_importance[fid]
            else:
                self.feature_importances_[fid] += tree_feature_importance[fid]
        LOGGER.debug('cur feature importance {}'.format(self.feature_importances_))

    def goss_sample(self):

        sampled_gh = goss_sampling(self.grad_and_hess, self.top_rate, self.other_rate)
        return sampled_gh

    def on_epoch_prepare(self, epoch_idx):
        """

        Parameters
        ----------
        epoch_idx cur epoch idx

        Returns None
        -------

        Prepare g, h, sample weights, sampling at the beginning of every epoch

        """
        if self.cur_epoch_idx != epoch_idx:
            self.grad_and_hess = self.compute_grad_and_hess(self.y_hat, self.y, self.data_inst)
            self.cur_epoch_idx = epoch_idx
            # goss sampling
            if self.enable_goss:
                if not self.re_compute_goss_sample_weight:
                    self.max_sample_weight = self.max_sample_weight * ((1 - self.top_rate) / self.other_rate)
                self.grad_and_hess = self.goss_sample()

    def preprocess(self):
        if self.multi_mode == consts.MULTI_OUTPUT:
            # re-set dimension
            self.booster_dim = 1

    def postprocess(self):
        host_feature_importance_list = self.hetero_sbt_transfer_variable.host_feature_importance.get(idx=-1)
        for i in host_feature_importance_list:
            self.feature_importances_.update(i)

        LOGGER.debug('self feature importance is {}'.format(self.feature_importances_))

    def fit_a_learner(self, epoch_idx: int, booster_dim: int):

        self.on_epoch_prepare(epoch_idx)

        if self.multi_mode == consts.MULTI_OUTPUT:
            g_h = self.grad_and_hess
        else:
            g_h = self.get_grad_and_hess(self.grad_and_hess, booster_dim)

        flow_id = self.generate_flowid(epoch_idx, booster_dim)
        complete_secure = True if (epoch_idx == 0 and self.complete_secure) else False

        tree_type, target_host_id = None, None
        fast_sbt = (self.boosting_strategy == consts.MIX_TREE or self.boosting_strategy == consts.LAYERED_TREE)
        if fast_sbt:
            tree_type, target_host_id = self.get_tree_plan(epoch_idx)
            self.check_host_number(tree_type)

        tree = produce_hetero_tree_learner(role=self.role, tree_param=self.tree_param, flow_id=flow_id,
                                           data_bin=self.data_bin, bin_split_points=self.bin_split_points,
                                           bin_sparse_points=self.bin_sparse_points, task_type=self.task_type,
                                           valid_features=self.sample_valid_features(),
                                           host_party_list=self.component_properties.host_party_idlist,
                                           runtime_idx=self.component_properties.local_partyid,
                                           cipher_compress=self.cipher_compressing,
                                           g_h=g_h, encrypter=self.encrypter,
                                           goss_subsample=self.enable_goss,
                                           complete_secure=complete_secure, max_sample_weights=self.max_sample_weight,
                                           fast_sbt=fast_sbt, tree_type=tree_type, target_host_id=target_host_id,
                                           guest_depth=self.guest_depth, host_depth=self.host_depth,
                                           mo_tree=(self.multi_mode == consts.MULTI_OUTPUT),
                                           class_num=len(self.classes_) if len(self.classes_) > 2 else 1  # mo parameter
                                           )

        tree.fit()
        self.update_feature_importance(tree.get_feature_importance())

        return tree

    def load_learner(self, model_meta, model_param, epoch_idx, booster_idx):

        flow_id = self.generate_flowid(epoch_idx, booster_idx)
        runtime_idx = self.component_properties.local_partyid
        host_list = self.component_properties.host_party_idlist
        fast_sbt = (self.boosting_strategy == consts.MIX_TREE or self.boosting_strategy == consts.LAYERED_TREE)
        tree_type, target_host_id = None, None

        if fast_sbt:
            tree_type, target_host_id = self.get_tree_plan(epoch_idx)

        tree = load_hetero_tree_learner(role=self.role, tree_param=self.tree_param, model_meta=model_meta,
                                        model_param=model_param,
                                        flow_id=flow_id, runtime_idx=runtime_idx, host_party_list=host_list,
                                        fast_sbt=fast_sbt, tree_type=tree_type, target_host_id=target_host_id)

        return tree

    def generate_summary(self) -> dict:

        summary = {'loss_history': self.history_loss,
                   'best_iteration': self.callback_variables.best_iteration,
                   'feature_importance': self.make_readable_feature_importance(self.feature_name_fid_mapping,
                                                                               self.feature_importances_),
                   'validation_metrics': self.callback_variables.validation_summary,
                   'is_converged': self.is_converged}

        return summary

    @staticmethod
    def make_readable_feature_importance(fid_mapping, feature_importances):
        """
        replace feature id by real feature name
        """
        new_fi = {}
        for id_ in feature_importances:
            if isinstance(id_, tuple):
                if consts.GUEST in id_[0]:
                    new_fi[fid_mapping[id_[1]]] = feature_importances[id_].importance
                else:
                    role, party_id = id_[0].split(':')
                    new_fi[generate_anonymous(role=role, fid=id_[1], party_id=party_id)] = feature_importances[
                        id_].importance
            else:
                new_fi[fid_mapping[id_]] = feature_importances[id_].importance

        return new_fi

    @assert_io_num_rows_equal
    def predict(self, data_inst, ret_format='std'):

        # standard format, leaf indices, raw score
        assert ret_format in ['std', 'leaf', 'raw'], 'illegal ret format'

        LOGGER.info('running prediction')
        cache_dataset_key = self.predict_data_cache.get_data_key(data_inst)

        processed_data = self.data_and_header_alignment(data_inst)

        last_round = self.predict_data_cache.predict_data_last_round(cache_dataset_key)

        self.sync_predict_round(last_round)

        rounds = len(self.boosting_model_list) // self.booster_dim
        trees = []
        LOGGER.debug('round involved in prediction {}, last round is {}, data key {}'
                     .format(list(range(last_round, rounds)), last_round, cache_dataset_key))

        for idx in range(last_round, rounds):
            for booster_idx in range(self.booster_dim):
                tree = self.load_learner(self.booster_meta,
                                         self.boosting_model_list[idx * self.booster_dim + booster_idx],
                                         idx, booster_idx)
                trees.append(tree)

        predict_cache = None
        tree_num = len(trees)

        if last_round != 0:
            predict_cache = self.predict_data_cache.predict_data_at(cache_dataset_key, min(rounds, last_round))
            LOGGER.info('load predict cache of round {}'.format(min(rounds, last_round)))

        if tree_num == 0 and predict_cache is not None and not (ret_format == 'leaf'):
            return self.score_to_predict_result(data_inst, predict_cache)

        if self.boosting_strategy == consts.MIX_TREE:
            predict_rs = mix_sbt_guest_predict(
                processed_data,
                self.hetero_sbt_transfer_variable,
                trees,
                self.learning_rate,
                self.init_score,
                self.booster_dim,
                predict_cache,
                pred_leaf=(
                    ret_format == 'leaf'))
        else:
            if self.EINI_inference and not self.on_training:  # EINI is for inference stage
                sitename = self.role + ':' + str(self.component_properties.local_partyid)
                predict_rs = EINI_guest_predict(
                    processed_data,
                    trees,
                    self.learning_rate,
                    self.init_score,
                    self.booster_dim,
                    self.encrypt_param.key_length,
                    self.hetero_sbt_transfer_variable,
                    sitename,
                    self.component_properties.host_party_idlist,
                    predict_cache,
                    False)
            else:
                predict_rs = sbt_guest_predict(
                    processed_data,
                    self.hetero_sbt_transfer_variable,
                    trees,
                    self.learning_rate,
                    self.init_score,
                    self.booster_dim,
                    predict_cache,
                    pred_leaf=(
                        ret_format == 'leaf'))

        if ret_format == 'leaf':
            return predict_rs  # predict result is leaf position

        self.predict_data_cache.add_data(cache_dataset_key, predict_rs, cur_boosting_round=rounds)
        LOGGER.debug('adding predict rs {}'.format(predict_rs))
        LOGGER.debug('last round is {}'.format(self.predict_data_cache.predict_data_last_round(cache_dataset_key)))

        if ret_format == 'raw':
            return predict_rs
        else:
            return self.score_to_predict_result(data_inst, predict_rs)

    def load_feature_importance(self, feat_importance_param):
        param = list(feat_importance_param)
        rs_dict = {}
        for fp in param:
            key = (fp.sitename, fp.fid)
            importance = FeatureImportance()
            importance.from_protobuf(fp)
            rs_dict[key] = importance

        self.feature_importances_ = rs_dict

    def get_model_meta(self):
        model_meta = BoostingTreeModelMeta()
        model_meta.tree_meta.CopyFrom(self.booster_meta)
        model_meta.learning_rate = self.learning_rate
        model_meta.num_trees = self.boosting_round
        model_meta.quantile_meta.CopyFrom(QuantileMeta(bin_num=self.bin_num))
        model_meta.objective_meta.CopyFrom(ObjectiveMeta(objective=self.objective_param.objective,
                                                         param=self.objective_param.params))
        model_meta.use_missing = self.use_missing
        model_meta.zero_as_missing = self.zero_as_missing
        model_meta.task_type = self.task_type
        model_meta.n_iter_no_change = self.n_iter_no_change
        model_meta.tol = self.tol
        model_meta.boosting_strategy = self.boosting_strategy
        model_meta.module = "HeteroSecureBoost"
        meta_name = consts.HETERO_SBT_GUEST_MODEL + "Meta"

        return meta_name, model_meta

    def get_model_param(self):

        model_param = BoostingTreeModelParam()
        model_param.tree_num = len(self.boosting_model_list)
        model_param.tree_dim = self.booster_dim
        model_param.trees_.extend(self.boosting_model_list)
        model_param.init_score.extend(self.init_score)
        model_param.losses.extend(self.history_loss)
        model_param.classes_.extend(map(str, self.classes_))
        model_param.num_classes = self.num_classes
        if self.boosting_strategy == consts.STD_TREE:
            model_param.model_name = consts.HETERO_SBT
        elif self.boosting_strategy == consts.LAYERED_TREE:
            model_param.model_name = consts.HETERO_FAST_SBT_LAYERED
        elif self.boosting_strategy == consts.MIX_TREE:
            model_param.model_name = consts.HETERO_FAST_SBT_MIX
        model_param.best_iteration = self.callback_variables.best_iteration

        feature_importances = list(self.feature_importances_.items())
        feature_importances = sorted(feature_importances, key=itemgetter(1), reverse=True)
        feature_importance_param = []

        for (sitename, fid), importance in feature_importances:
            if consts.GUEST in sitename:
                fullname = self.feature_name_fid_mapping[fid]
            else:
                role_name, party_id = sitename.split(':')
                fullname = generate_anonymous(fid=fid, party_id=party_id, role=role_name)

            feature_importance_param.append(FeatureImportanceInfo(sitename=sitename,  # sitename to distinguish sites
                                                                  fid=fid,
                                                                  importance=importance.importance,
                                                                  fullname=fullname,
                                                                  importance2=importance.importance_2,
                                                                  main=importance.main_type
                                                                  ))
        model_param.feature_importances.extend(feature_importance_param)
        model_param.feature_name_fid_mapping.update(self.feature_name_fid_mapping)
        model_param.tree_plan.extend(plan.encode_plan(self.tree_plan))
        param_name = consts.HETERO_SBT_GUEST_MODEL + "Param"

        return param_name, model_param

    def set_model_meta(self, model_meta):

        if not self.is_warm_start:
            # these hyper parameters are not needed in warm start setting
            self.boosting_round = model_meta.num_trees
            self.tol = model_meta.tol
            self.n_iter_no_change = model_meta.n_iter_no_change
            self.bin_num = model_meta.quantile_meta.bin_num

        self.learning_rate = model_meta.learning_rate
        self.booster_meta = model_meta.tree_meta
        self.objective_param.objective = model_meta.objective_meta.objective
        self.objective_param.params = list(model_meta.objective_meta.param)
        self.task_type = model_meta.task_type
        self.boosting_strategy = model_meta.boosting_strategy

    def set_model_param(self, model_param):

        self.boosting_model_list = list(model_param.trees_)
        self.init_score = np.array(list(model_param.init_score))
        self.history_loss = list(model_param.losses)
        self.classes_ = list(map(int, model_param.classes_))
        self.booster_dim = model_param.tree_dim
        self.num_classes = model_param.num_classes
        self.feature_name_fid_mapping.update(model_param.feature_name_fid_mapping)
        self.load_feature_importance(model_param.feature_importances)
        # initialize loss function
        self.loss = self.get_loss_function()
        # init model tree plan if it exists
        self.tree_plan = plan.decode_plan(model_param.tree_plan)
