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

from federatedml.param.base_param import BaseParam
from federatedml.param.encrypt_param import EncryptParam
from federatedml.param.encrypted_mode_calculation_param import EncryptedModeCalculatorParam
from federatedml.param.cross_validation_param import CrossValidationParam
from federatedml.param.predict_param import PredictParam
from federatedml.util import consts
import copy
import collections

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class ObjectiveParam(BaseParam):
    """
    Define objective parameters that used in federated ml.

    Parameters
    ----------
    objective : None or str, accepted None,'cross_entropy','lse','lae','log_cosh','tweedie','fair','huber' only,
                None in host's config, should be str in guest'config.
                when task_type is classification, only support cross_enctropy,
                other 6 types support in regression task. default: None

    params : None or list, should be non empty list when objective is 'tweedie','fair','huber',
             first element of list shoulf be a float-number large than 0.0 when objective is 'fair','huber',
             first element of list should be a float-number in [1.0, 2.0) when objective is 'tweedie'
    """

    def __init__(self, objective=None, params=None):
        self.objective = objective
        self.params = params

    def check(self, task_type=None):
        if self.objective is None:
            return True

        descr = "objective param's"

        LOGGER.debug('check objective {}'.format(self.objective))

        if task_type not in [consts.CLASSIFICATION, consts.REGRESSION]:
            self.objective = self.check_and_change_lower(self.objective,
                                                   ["cross_entropy", "lse", "lae", "huber", "fair",
                                                    "log_cosh", "tweedie"],
                                                       descr)

        if task_type == consts.CLASSIFICATION:
            if self.objective != "cross_entropy":
                raise ValueError("objective param's objective {} not supported".format(self.objective))

        elif task_type == consts.REGRESSION:
            self.objective = self.check_and_change_lower(self.objective,
                                                               ["lse", "lae", "huber", "fair", "log_cosh", "tweedie"],
                                                               descr)

            params = self.params
            if self.objective in ["huber", "fair", "tweedie"]:
                if type(params).__name__ != 'list' or len(params) < 1:
                    raise ValueError(
                        "objective param's params {} not supported, should be non-empty list".format(params))

                if type(params[0]).__name__ not in ["float", "int", "long"]:
                    raise ValueError("objective param's params[0] {} not supported".format(self.params[0]))

                if self.objective == 'tweedie':
                    if params[0] < 1 or params[0] >= 2:
                        raise ValueError("in tweedie regression, objective params[0] should betweend [1, 2)")

                if self.objective == 'fair' or 'huber':
                    if params[0] <= 0.0:
                        raise ValueError("in {} regression, objective params[0] should greater than 0.0".format(
                            self.objective))
        return True


class DecisionTreeParam(BaseParam):
    """
    Define decision tree parameters that used in federated ml.

    Parameters
    ----------
    criterion_method : str, accepted "xgboost" only, the criterion function to use, default: 'xgboost'

    criterion_params: list, should be non empty and first element is float-number, default: 0.1.

    max_depth: int, positive integer, the max depth of a decision tree, default: 5

    min_sample_split: int, least quantity of nodes to split, default: 2

    min_impurity_split: float, least gain of a single split need to reach, default: 1e-3

    min_leaf_node: int, when samples no more than min_leaf_node, it becomes a leave, default: 1

    max_split_nodes: int, positive integer, we will use no more than max_split_nodes to
                      parallel finding their splits in a batch, for memory consideration. default is 65536

    n_iter_no_change: bool, accepted True,False only, if set to True, tol will use to consider
                      stop tree growth. default: True

    feature_importance_type: str, support 'split', 'gain' only.
                             if is 'split', feature_importances calculate by feature split times,
                             if is 'gain', feature_importances calculate by feature split gain.
                             default: 'split'

    tol: float, only use when n_iter_no_change is set to True, default: 0.001

    use_missing: bool, accepted True, False only, use missing value in training process or not. default: False

    zero_as_missing: bool, accepted True, False only, regard 0 as missing value or not,
                     will be use only if use_missing=True, default: False

    """

    def __init__(self, criterion_method="xgboost", criterion_params=[0.1], max_depth=5,
                 min_sample_split=2, min_imputiry_split=1e-3, min_leaf_node=1,
                 max_split_nodes=consts.MAX_SPLIT_NODES, feature_importance_type="split",
                 n_iter_no_change=True, tol=0.001,
                 use_missing=False, zero_as_missing=False,):

        self.criterion_method = criterion_method
        self.criterion_params = criterion_params
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_impurity_split = min_imputiry_split
        self.min_leaf_node = min_leaf_node
        self.max_split_nodes = max_split_nodes
        self.feature_importance_type = feature_importance_type
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.use_missing = use_missing
        self.zero_as_missing = zero_as_missing

    def check(self):
        descr = "decision tree param"

        self.criterion_method = self.check_and_change_lower(self.criterion_method,
                                                             ["xgboost"],
                                                             descr)

        if type(self.criterion_params).__name__ != "list":
            raise ValueError("decision tree param's criterion_params {} not supported, should be list".format(
                self.criterion_params))

        if len(self.criterion_params) == 0:
            raise ValueError("decisition tree param's criterio_params should be non empty")

        if type(self.criterion_params[0]).__name__ not in ["int", "long", "float"]:
            raise ValueError("decision tree param's criterion_params element shoubld be numeric")

        if type(self.max_depth).__name__ not in ["int", "long"]:
            raise ValueError("decision tree param's max_depth {} not supported, should be integer".format(
                self.max_depth))

        if self.max_depth < 1:
            raise ValueError("decision tree param's max_depth should be positive integer, no less than 1")

        if type(self.min_sample_split).__name__ not in ["int", "long"]:
            raise ValueError("decision tree param's min_sample_split {} not supported, should be integer".format(
                self.min_sample_split))

        if type(self.min_impurity_split).__name__ not in ["int", "long", "float"]:
            raise ValueError("decision tree param's min_impurity_split {} not supported, should be numeric".format(
                self.min_impurity_split))

        if type(self.min_leaf_node).__name__ not in ["int", "long"]:
            raise ValueError("decision tree param's min_leaf_node {} not supported, should be integer".format(
                self.min_leaf_node))

        if type(self.max_split_nodes).__name__ not in ["int", "long"] or self.max_split_nodes < 1:
            raise ValueError("decision tree param's max_split_nodes {} not supported, " + \
                             "should be positive integer between 1 and {}".format(self.max_split_nodes,
                                                                                  consts.MAX_SPLIT_NODES))

        if type(self.n_iter_no_change).__name__ != "bool":
            raise ValueError("decision tree param's n_iter_no_change {} not supported, should be bool type".format(
                self.n_iter_no_change))

        if type(self.tol).__name__ not in ["float", "int", "long"]:
            raise ValueError("decision tree param's tol {} not supported, should be numeric".format(self.tol))

        self.feature_importance_type = self.check_and_change_lower(self.feature_importance_type,
                                                                    ["split", "gain"],
                                                                    descr)

        return True


class BoostingParam(BaseParam):
    """
        Basic parameter for Boosting Algorithms

        Parameters
        ----------
        task_type : str, accepted 'classification', 'regression' only, default: 'classification'

        objective_param : ObjectiveParam Object, default: ObjectiveParam()

        learning_rate : float, accepted float, int or long only, the learning rate of secure boost. default: 0.3

        num_trees : int, accepted int, float only, the max number of boosting round. default: 5

        subsample_feature_rate : float, a float-number in [0, 1], default: 0.8

        n_iter_no_change : bool,
            when True and residual error less than tol, tree building process will stop. default: True

        bin_num: int, positive integer greater than 1, bin number use in quantile. default: 32

        validation_freqs: None or positive integer or container object in python. Do validation in training process or Not.
                          if equals None, will not do validation in train process;
                          if equals positive integer, will validate data every validation_freqs epochs passes;
                          if container object in python, will validate data if epochs belong to this container.
                            e.g. validation_freqs = [10, 15], will validate data when epoch equals to 10 and 15.
                          Default: None
        """

    def __init__(self,  task_type=consts.CLASSIFICATION,
                 objective_param=ObjectiveParam(),
                 learning_rate=0.3, num_trees=5, subsample_feature_rate=0.8, n_iter_no_change=True,
                 tol=0.0001, bin_num=32,
                 predict_param=PredictParam(), cv_param=CrossValidationParam(),
                 validation_freqs=None, metrics=None):

        super(BoostingParam, self).__init__()

        self.task_type = task_type
        self.objective_param = copy.deepcopy(objective_param)
        self.learning_rate = learning_rate
        self.num_trees = num_trees
        self.subsample_feature_rate = subsample_feature_rate
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.bin_num = bin_num
        self.predict_param = copy.deepcopy(predict_param)
        self.cv_param = copy.deepcopy(cv_param)
        self.validation_freqs = validation_freqs
        self.metrics = metrics

    def check(self):

        descr = "boosting_core tree param's"

        if self.task_type not in [consts.CLASSIFICATION, consts.REGRESSION]:
            raise ValueError("boosting_core tree param's task_type {} not supported, should be {} or {}".format(
                self.task_type, consts.CLASSIFICATION, consts.REGRESSION))

        self.objective_param.check(self.task_type)

        if type(self.learning_rate).__name__ not in ["float", "int", "long"]:
            raise ValueError("boosting_core tree param's learning_rate {} not supported, should be numeric".format(
                self.learning_rate))

        if type(self.subsample_feature_rate).__name__ not in ["float", "int", "long"] or \
                self.subsample_feature_rate < 0 or self.subsample_feature_rate > 1:
            raise ValueError("boosting_core tree param's subsample_feature_rate should be a numeric number between 0 and 1")

        if type(self.n_iter_no_change).__name__ != "bool":
            raise ValueError("boosting_core tree param's n_iter_no_change {} not supported, should be bool type".format(
                self.n_iter_no_change))

        if type(self.tol).__name__ not in ["float", "int", "long"]:
            raise ValueError("boosting_core tree param's tol {} not supported, should be numeric".format(self.tol))

        if type(self.bin_num).__name__ not in ["int", "long"] or self.bin_num < 2:
            raise ValueError(
                "boosting_core tree param's bin_num {} not supported, should be positive integer greater than 1".format(
                    self.bin_num))

        if self.validation_freqs is None:
            pass
        elif isinstance(self.validation_freqs, int):
            if self.validation_freqs < 1:
                raise ValueError("validation_freqs should be larger than 0 when it's integer")
        elif not isinstance(self.validation_freqs, collections.Container):
            raise ValueError("validation_freqs should be None or positive integer or container")

        if self.metrics is not None and not isinstance(self.metrics, list):
            raise ValueError("metrics should be a list")

        return True


class HeteroBoostingParam(BoostingParam):

    """
    encrypt_param : EncodeParam Object, encrypt method use in secure boost, default: EncryptParam()

    encrypted_mode_calculator_param: EncryptedModeCalculatorParam object, the calculation mode use in secureboost,
                                     default: EncryptedModeCalculatorParam()
    """

    def __init__(self, task_type=consts.CLASSIFICATION,
                 objective_param=ObjectiveParam(),
                 learning_rate=0.3, num_trees=5, subsample_feature_rate=0.8, n_iter_no_change=True,
                 tol=0.0001, encrypt_param=EncryptParam(),
                 bin_num=32,
                 encrypted_mode_calculator_param=EncryptedModeCalculatorParam(),
                 predict_param=PredictParam(), cv_param=CrossValidationParam(),
                 validation_freqs=None, early_stopping_rounds=None, metrics=None, use_first_metric_only=False):

        super(HeteroBoostingParam, self).__init__(task_type, objective_param, learning_rate, num_trees,
                                                  subsample_feature_rate, n_iter_no_change, tol, bin_num,
                                                  predict_param, cv_param, validation_freqs, metrics=metrics)

        self.encrypt_param = copy.deepcopy(encrypt_param)
        self.encrypted_mode_calculator_param = copy.deepcopy(encrypted_mode_calculator_param)
        self.early_stopping_rounds = early_stopping_rounds
        self.use_first_metric_only = use_first_metric_only

    def check(self):

        super(HeteroBoostingParam, self).check()
        self.encrypted_mode_calculator_param.check()
        self.encrypt_param.check()

        if self.early_stopping_rounds is None:
            pass
        elif isinstance(self.early_stopping_rounds, int):
            if self.early_stopping_rounds < 1:
                raise ValueError("early stopping rounds should be larger than 0 when it's integer")
            if self.validation_freqs is None:
                raise ValueError("validation freqs must be set when early stopping is enabled")

        if not isinstance(self.use_first_metric_only, bool):
            raise ValueError("use_first_metric_only should be a boolean")

        return True


class HeteroSecureBoostParam(HeteroBoostingParam):
    """
        Define boosting tree parameters that used in federated ml.

        Parameters
        ----------
        task_type : str, accepted 'classification', 'regression' only, default: 'classification'

        tree_param : DecisionTreeParam Object, default: DecisionTreeParam()

        objective_param : ObjectiveParam Object, default: ObjectiveParam()

        learning_rate : float, accepted float, int or long only, the learning rate of secure boost. default: 0.3

        num_trees : int, accepted int, float only, the max number of trees to build. default: 5

        subsample_feature_rate : float, a float-number in [0, 1], default: 0.8

        n_iter_no_change : bool,
            when True and residual error less than tol, tree building process will stop. default: True

        encrypt_param : EncodeParam Object, encrypt method use in secure boost, default: EncryptParam(), this parameter
                        is only for hetero-secureboost

        bin_num: int, positive integer greater than 1, bin number use in quantile. default: 32

        encrypted_mode_calculator_param: EncryptedModeCalculatorParam object, the calculation mode use in secureboost,
                                         default: EncryptedModeCalculatorParam(), only for hetero-secureboost

        use_missing: bool, accepted True, False only, use missing value in training process or not. default: False

        zero_as_missing: bool, accepted True, False only, regard 0 as missing value or not,
                         will be use only if use_missing=True, default: False

        validation_freqs: None or positive integer or container object in python. Do validation in training process or Not.
                          if equals None, will not do validation in train process;
                          if equals positive integer, will validate data every validation_freqs epochs passes;
                          if container object in python, will validate data if epochs belong to this container.
                            e.g. validation_freqs = [10, 15], will validate data when epoch equals to 10 and 15.
                          Default: None
                          The default value is None, 1 is suggested. You can set it to a number larger than 1 in order to
                          speed up training by skipping validation rounds. When it is larger than 1, a number which is
                          divisible by "num_trees" is recommended, otherwise, you will miss the validation scores
                          of last training iteration.

        early_stopping_rounds: should be a integer larger than 0，will stop training if one metric of one validation data
                                doesn’t improve in last early_stopping_round rounds，
                                need to set validation freqs and will check early_stopping every at every validation epoch,

        metrics: list, default: []
                 Specify which metrics to be used when performing evaluation during training process.
                 If set as empty, default metrics will be used. For regression tasks, default metrics are
                 ['root_mean_squared_error', 'mean_absolute_error']， For binary-classificatiin tasks, default metrics
                 are ['auc', 'ks']. For multi-classification tasks, default metrics are ['accuracy', 'precision', 'recall']

        use_first_metric_only: use only the first metric for early stopping

        complete_secure: bool, if use complete_secure, when use complete secure, build first tree using only guest
                        features
        """

    def __init__(self, tree_param: DecisionTreeParam = DecisionTreeParam(), task_type=consts.CLASSIFICATION,
                 objective_param=ObjectiveParam(),
                 learning_rate=0.3, num_trees=5, subsample_feature_rate=0.8, n_iter_no_change=True,
                 tol=0.0001, encrypt_param=EncryptParam(),
                 bin_num=32,
                 encrypted_mode_calculator_param=EncryptedModeCalculatorParam(),
                 predict_param=PredictParam(), cv_param=CrossValidationParam(),
                 validation_freqs=None, early_stopping_rounds=None, use_missing=False, zero_as_missing=False,
                 complete_secure=False, metrics=None, use_first_metric_only=False):

        super(HeteroSecureBoostParam, self).__init__(task_type, objective_param, learning_rate, num_trees,
                                                     subsample_feature_rate, n_iter_no_change, tol, encrypt_param,
                                                     bin_num, encrypted_mode_calculator_param, predict_param, cv_param,
                                                     validation_freqs, early_stopping_rounds, metrics=metrics,
                                                     use_first_metric_only=use_first_metric_only)

        self.tree_param = tree_param
        self.zero_as_missing = zero_as_missing
        self.use_missing = use_missing
        self.complete_secure = complete_secure

    def check(self):

        super(HeteroSecureBoostParam, self).check()
        self.tree_param.check()
        if type(self.use_missing) != bool:
            raise ValueError('use missing should be bool type')
        if type(self.zero_as_missing) != bool:
            raise ValueError('zero as missing should be bool type')

        return True


class HeteroFastSecureBoostParam(HeteroSecureBoostParam):

    def __init__(self, tree_param: DecisionTreeParam = DecisionTreeParam(), task_type=consts.CLASSIFICATION,
                 objective_param=ObjectiveParam(),
                 learning_rate=0.3, num_trees=5, subsample_feature_rate=0.8, n_iter_no_change=True,
                 tol=0.0001, encrypt_param=EncryptParam(),
                 bin_num=32,
                 encrypted_mode_calculator_param=EncryptedModeCalculatorParam(),
                 predict_param=PredictParam(), cv_param=CrossValidationParam(),
                 validation_freqs=None, early_stopping=None, use_missing=False, zero_as_missing=False,
                 complete_secure=False, k=1, guest_depth=1, host_depth=1, work_mode='mix', metrics=None):

        """
        work_mode：
            mix:  alternate using guest/host features to build trees. For example, the first k trees use guest features,
                  the second k trees use host features, and so on
            layered: only support 2 party, when running layered mode, first 'host_depth' layer will use host features,
                     and then next 'guest_depth' will only use guest features
        k: every party will alternate build k trees until reach max tree num, this param is valid when work_mode is
            mix
        guest_depth: guest will build last guest_depth of a decision tree using guest features, is valid when work mode
            is layered
        host depth: host will build first host_depth of a decision tree using host features, is valid when work mode is
            layered

        other params are the same as HeteroSecureBoost

        """

        super(HeteroFastSecureBoostParam, self).__init__(tree_param, task_type, objective_param, learning_rate,
                                                         num_trees, subsample_feature_rate, n_iter_no_change, tol,
                                                         encrypt_param, bin_num, encrypted_mode_calculator_param,
                                                         predict_param, cv_param, validation_freqs, early_stopping,
                                                         use_missing, zero_as_missing, complete_secure, metrics=metrics)

        self.k = k
        self.guest_depth = guest_depth
        self.host_depth = host_depth
        self.work_mode = work_mode

    def check(self):

        super(HeteroFastSecureBoostParam, self).check()
        if type(self.guest_depth).__name__ not in ["int", "long"] or self.guest_depth <= 0:
            raise ValueError("guest_depth should be larger than 0")
        if type(self.host_depth).__name__ not in ["int", "long"] or self.guest_depth <= 0:
            raise ValueError("host_depth should be larger than 0")
        if type(self.k).__name__ not in ["int", "long"] or self.guest_depth <= 0:
            raise ValueError("k should be larger than 0")

        work_modes = [consts.MIX_TREE, consts.LAYERED_TREE]
        if self.work_mode not in work_modes:
            raise ValueError('only work_modes: {} are supported, input work mode is {}'.
                             format(work_modes, self.work_mode))

        return True


class HomoSecureBoostParam(BoostingParam):

    def __init__(self, tree_param: DecisionTreeParam = DecisionTreeParam(), task_type=consts.CLASSIFICATION,
                 objective_param=ObjectiveParam(),
                 learning_rate=0.3, num_trees=5, subsample_feature_rate=0.8, n_iter_no_change=True,
                 tol=0.0001,
                 bin_num=32,
                 predict_param=PredictParam(), cv_param=CrossValidationParam(),
                 validation_freqs=None, use_missing=False, zero_as_missing=False,
                 ):
        super(HomoSecureBoostParam, self).__init__(task_type=task_type,
                                                   objective_param=objective_param,
                                                   learning_rate=learning_rate,
                                                   num_trees=num_trees,
                                                   subsample_feature_rate=subsample_feature_rate,
                                                   n_iter_no_change=n_iter_no_change,
                                                   tol=tol,
                                                   bin_num=bin_num,
                                                   predict_param=predict_param,
                                                   cv_param=cv_param,
                                                   validation_freqs=validation_freqs,
                                                   )
        self.use_missing = use_missing
        self.zero_as_missing = zero_as_missing
        self.tree_param = tree_param

    def check(self):
        super(HomoSecureBoostParam, self).check()
        self.tree_param.check()
        if type(self.use_missing) != bool:
            raise ValueError('use missing should be bool type')
        if type(self.zero_as_missing) != bool:
            raise ValueError('zero as missing should be bool type')
        return True








