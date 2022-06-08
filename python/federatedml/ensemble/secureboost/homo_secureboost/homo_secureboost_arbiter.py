import numpy as np
from numpy import random
from federatedml.util import LOGGER
from federatedml.util import consts
from federatedml.ensemble.boosting.homo_boosting import HomoBoostingArbiter
from federatedml.param.boosting_param import HomoSecureBoostParam
from federatedml.ensemble.basic_algorithms.decision_tree.homo.homo_decision_tree_arbiter import HomoDecisionTreeArbiter


class HomoSecureBoostingTreeArbiter(HomoBoostingArbiter):

    def __init__(self):
        super(HomoSecureBoostingTreeArbiter, self).__init__()
        self.model_name = 'HomoSecureBoost'
        self.tree_param = None  # decision tree param
        self.use_missing = False
        self.zero_as_missing = False
        self.cur_epoch_idx = -1
        self.grad_and_hess = None
        self.feature_importances_ = {}
        self.model_param = HomoSecureBoostParam()

        self.multi_mode = consts.SINGLE_OUTPUT

    def _init_model(self, boosting_param: HomoSecureBoostParam):
        super(HomoSecureBoostingTreeArbiter, self)._init_model(boosting_param)
        self.use_missing = boosting_param.use_missing
        self.zero_as_missing = boosting_param.zero_as_missing
        self.tree_param = boosting_param.tree_param
        self.multi_mode = boosting_param.multi_mode
        if self.use_missing:
            self.tree_param.use_missing = self.use_missing
            self.tree_param.zero_as_missing = self.zero_as_missing

    def send_valid_features(self, valid_features, epoch_idx, b_idx):
        self.transfer_inst.valid_features.remote(valid_features, idx=-1, suffix=('valid_features', epoch_idx, b_idx))

    def sample_valid_features(self):

        LOGGER.info("sample valid features")
        chosen_feature = random.choice(range(0, self.feature_num),
                                       max(1, int(self.subsample_feature_rate * self.feature_num)), replace=False)
        valid_features = [False for i in range(self.feature_num)]
        for fid in chosen_feature:
            valid_features[fid] = True

        return valid_features

    def preprocess(self):
        if self.multi_mode == consts.MULTI_OUTPUT:
            self.booster_dim = 1

    def fit_a_learner(self, epoch_idx: int, booster_dim: int):

        valid_feature = self.sample_valid_features()
        self.send_valid_features(valid_feature, epoch_idx, booster_dim)
        flow_id = self.generate_flowid(epoch_idx, booster_dim)
        new_tree = HomoDecisionTreeArbiter(self.tree_param, valid_feature=valid_feature, epoch_idx=epoch_idx,
                                           flow_id=flow_id, tree_idx=booster_dim)
        new_tree.fit()

        return new_tree

    def generate_summary(self) -> dict:

        summary = {'loss_history': self.history_loss}
        return summary

    # homo tree arbiter doesnt save model
    def get_cur_model(self):
        return None

    def load_learner(self, model_meta, model_param, epoch_idx, booster_idx):
        pass

    def set_model_param(self, model_param):
        pass

    def set_model_meta(self, model_meta):
        pass

    def get_model_param(self):
        pass

    def get_model_meta(self):
        pass
