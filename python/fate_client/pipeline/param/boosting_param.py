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

from pipeline.param.base_param import BaseParam
from pipeline.param.encrypt_param import EncryptParam
from pipeline.param.encrypted_mode_calculation_param import EncryptedModeCalculatorParam
from pipeline.param.cross_validation_param import CrossValidationParam
from pipeline.param.predict_param import PredictParam
from pipeline.param import consts
from pipeline.param.callback_param import CallbackParam
import copy
import collections


class ObjectiveParam(BaseParam):
    """
    Define objective parameters that used in federated ml.

    Parameters
    ----------
    objective : {None, 'cross_entropy', 'lse', 'lae', 'log_cosh', 'tweedie', 'fair', 'huber'}
        None in host's config, should be str in guest'config.
        when task_type is classification, only support 'cross_entropy',
        other 6 types support in regression task

    params : None or list
        should be non empty list when objective is 'tweedie','fair','huber',
        first element of list shoulf be a float-number large than 0.0 when objective is 'fair', 'huber',
        first element of list should be a float-number in [1.0, 2.0) when objective is 'tweedie'
    """

    def __init__(self, objective='cross_entropy', params=None):
        self.objective = objective
        self.params = params

    def check(self, task_type=None):
        if self.objective is None:
            return True

        descr = "objective param's"

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
    criterion_method : {"xgboost"}, default: "xgboost"
        the criterion function to use

    criterion_params: list or dict
        should be non empty and elements are float-numbers,
        if a list is offered, the first one is l2 regularization value, and the second one is
        l1 regularization value.
        if a dict is offered, make sure it contains key 'l1', and 'l2'.
        l1, l2 regularization values are non-negative floats.
        default: [0.1, 0] or {'l1':0, 'l2':0,1}

    max_depth: positive integer
        the max depth of a decision tree, default: 3

    min_sample_split: int
        least quantity of nodes to split, default: 2

    min_impurity_split: float
        least gain of a single split need to reach, default: 1e-3

    min_child_weight: float
        sum of hessian needed in child nodes. default is 0

    min_leaf_node: int
        when samples no more than min_leaf_node, it becomes a leave, default: 1

    max_split_nodes: positive integer
        we will use no more than max_split_nodes to
        parallel finding their splits in a batch, for memory consideration. default is 65536

    feature_importance_type: {'split', 'gain'}
        if is 'split', feature_importances calculate by feature split times,
        if is 'gain', feature_importances calculate by feature split gain.
        default: 'split'

                             Due to the safety concern, we adjust training strategy of Hetero-SBT in FATE-1.8,
                             When running Hetero-SBT, this parameter is now abandoned.
                             In Hetero-SBT of FATE-1.8, guest side will compute split, gain of local features,
                             and receive anonymous feature importance results from hosts. Hosts will compute split
                             importance of local features.

    use_missing: bool, accepted True, False only, use missing value in training process or not. default: False

    zero_as_missing: bool
        regard 0 as missing value or not,
        will be use only if use_missing=True, default: False

    deterministic: bool
        ensure stability when computing histogram. Set this to true to ensure stable result when using
        same data and same parameter. But it may slow down computation.

    """

    def __init__(self, criterion_method="xgboost", criterion_params=[0.1, 0], max_depth=3,
                 min_sample_split=2, min_impurity_split=1e-3, min_leaf_node=1,
                 max_split_nodes=consts.MAX_SPLIT_NODES, feature_importance_type='split',
                 n_iter_no_change=True, tol=0.001, min_child_weight=0,
                 use_missing=False, zero_as_missing=False, deterministic=False):

        super(DecisionTreeParam, self).__init__()

        self.criterion_method = criterion_method
        self.criterion_params = criterion_params
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.min_impurity_split = min_impurity_split
        self.min_leaf_node = min_leaf_node
        self.min_child_weight = min_child_weight
        self.max_split_nodes = max_split_nodes
        self.feature_importance_type = feature_importance_type
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.use_missing = use_missing
        self.zero_as_missing = zero_as_missing
        self.deterministic = deterministic

    def check(self):
        descr = "decision tree param"

        self.criterion_method = self.check_and_change_lower(self.criterion_method,
                                                            ["xgboost"],
                                                            descr)

        if len(self.criterion_params) == 0:
            raise ValueError("decisition tree param's criterio_params should be non empty")

        if isinstance(self.criterion_params, list):
            assert len(self.criterion_params) == 2, 'length of criterion_param should be 2: l1, l2 regularization ' \
                                                    'values are needed'
            self.check_nonnegative_number(self.criterion_params[0], 'l2 reg value')
            self.check_nonnegative_number(self.criterion_params[1], 'l1 reg value')

        elif isinstance(self.criterion_params, dict):
            assert 'l1' in self.criterion_params and 'l2' in self.criterion_params, 'l1 and l2 keys are needed in ' \
                                                                                    'criterion_params dict'
            self.criterion_params = [self.criterion_params['l2'], self.criterion_params['l1']]
        else:
            raise ValueError('criterion_params should be a dict or a list contains l1, l2 reg value')

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
            raise ValueError("decision tree param's max_split_nodes {} not supported, " +
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
        self.check_nonnegative_number(self.min_child_weight, 'min_child_weight')
        self.check_boolean(self.deterministic, 'deterministic')

        return True


class BoostingParam(BaseParam):
    """
    Basic parameter for Boosting Algorithms

    Parameters
    ----------
    task_type : {'classification', 'regression'}, default: 'classification'
        task type

    objective_param : ObjectiveParam Object, default: ObjectiveParam()
        objective param

    learning_rate : float, int or long
        the learning rate of secure boost. default: 0.3

    num_trees : int or float
        the max number of boosting round. default: 5

    subsample_feature_rate : float
        a float-number in [0, 1], default: 1.0

    n_iter_no_change : bool,
        when True and residual error less than tol, tree building process will stop. default: True

    bin_num: positive integer greater than 1
        bin number use in quantile. default: 32

    validation_freqs: None or positive integer or container object in python
        Do validation in training process or Not.
        if equals None, will not do validation in train process;
        if equals positive integer, will validate data every validation_freqs epochs passes;
        if container object in python, will validate data if epochs belong to this container.
        e.g. validation_freqs = [10, 15], will validate data when epoch equals to 10 and 15.
        Default: None
        """

    def __init__(self, task_type=consts.CLASSIFICATION,
                 objective_param=ObjectiveParam(),
                 learning_rate=0.3, num_trees=5, subsample_feature_rate=1, n_iter_no_change=True,
                 tol=0.0001, bin_num=32,
                 predict_param=PredictParam(), cv_param=CrossValidationParam(),
                 validation_freqs=None, metrics=None, random_seed=100,
                 binning_error=consts.DEFAULT_RELATIVE_ERROR):

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
        self.random_seed = random_seed
        self.binning_error = binning_error

    def check(self):

        descr = "boosting tree param's"

        if self.task_type not in [consts.CLASSIFICATION, consts.REGRESSION]:
            raise ValueError("boosting_core tree param's task_type {} not supported, should be {} or {}".format(
                self.task_type, consts.CLASSIFICATION, consts.REGRESSION))

        self.objective_param.check(self.task_type)

        if type(self.learning_rate).__name__ not in ["float", "int", "long"]:
            raise ValueError("boosting_core tree param's learning_rate {} not supported, should be numeric".format(
                self.learning_rate))

        if type(self.subsample_feature_rate).__name__ not in ["float", "int", "long"] or \
                self.subsample_feature_rate < 0 or self.subsample_feature_rate > 1:
            raise ValueError(
                "boosting_core tree param's subsample_feature_rate should be a numeric number between 0 and 1")

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

        if self.random_seed is not None:
            assert isinstance(self.random_seed, int) and self.random_seed >= 0, 'random seed must be an integer >= 0'

        self.check_decimal_float(self.binning_error, descr)

        return True


class HeteroBoostingParam(BoostingParam):
    """
    Parameters
    ----------
    encrypt_param : EncodeParam Object
        encrypt method use in secure boost, default: EncryptParam()

    encrypted_mode_calculator_param: EncryptedModeCalculatorParam object
        the calculation mode use in secureboost,
        default: EncryptedModeCalculatorParam()
    """

    def __init__(self, task_type=consts.CLASSIFICATION,
                 objective_param=ObjectiveParam(),
                 learning_rate=0.3, num_trees=5, subsample_feature_rate=1, n_iter_no_change=True,
                 tol=0.0001, encrypt_param=EncryptParam(),
                 bin_num=32,
                 encrypted_mode_calculator_param=EncryptedModeCalculatorParam(),
                 predict_param=PredictParam(), cv_param=CrossValidationParam(),
                 validation_freqs=None, early_stopping_rounds=None, metrics=None, use_first_metric_only=False,
                 random_seed=100, binning_error=consts.DEFAULT_RELATIVE_ERROR):

        super(HeteroBoostingParam, self).__init__(task_type, objective_param, learning_rate, num_trees,
                                                  subsample_feature_rate, n_iter_no_change, tol, bin_num,
                                                  predict_param, cv_param, validation_freqs, metrics=metrics,
                                                  random_seed=random_seed,
                                                  binning_error=binning_error)

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
    task_type : {'classification', 'regression'}, default: 'classification'
        task type

    tree_param : DecisionTreeParam Object, default: DecisionTreeParam()
        tree param

    objective_param : ObjectiveParam Object, default: ObjectiveParam()
        objective param

    learning_rate : float, int or long
        the learning rate of secure boost. default: 0.3

    num_trees : int or float
        the max number of trees to build. default: 5

    subsample_feature_rate : float
        a float-number in [0, 1], default: 1.0

    random_seed: int
        seed that controls all random functions

    n_iter_no_change : bool,
        when True and residual error less than tol, tree building process will stop. default: True

    encrypt_param : EncodeParam Object
        encrypt method use in secure boost, default: EncryptParam(), this parameter
        is only for hetero-secureboost

    bin_num: positive integer greater than 1
        bin number use in quantile. default: 32

    encrypted_mode_calculator_param: EncryptedModeCalculatorParam object
        the calculation mode use in secureboost, default: EncryptedModeCalculatorParam(), only for hetero-secureboost

    use_missing: bool
        use missing value in training process or not. default: False

    zero_as_missing: bool
        regard 0 as missing value or not, will be use only if use_missing=True, default: False

    validation_freqs: None or positive integer or container object in python
        Do validation in training process or Not.
        if equals None, will not do validation in train process;
        if equals positive integer, will validate data every validation_freqs epochs passes;
        if container object in python, will validate data if epochs belong to this container.
        e.g. validation_freqs = [10, 15], will validate data when epoch equals to 10 and 15.
        Default: None
        The default value is None, 1 is suggested. You can set it to a number larger than 1 in order to
        speed up training by skipping validation rounds. When it is larger than 1, a number which is
        divisible by "num_trees" is recommended, otherwise, you will miss the validation scores
        of last training iteration.

    early_stopping_rounds: integer larger than 0
        will stop training if one metric of one validation data
        doesn’t improve in last early_stopping_round rounds，
        need to set validation freqs and will check early_stopping every at every validation epoch,

    metrics: list, default: []
        Specify which metrics to be used when performing evaluation during training process.
        If set as empty, default metrics will be used. For regression tasks, default metrics are
        ['root_mean_squared_error', 'mean_absolute_error']， For binary-classificatiin tasks, default metrics
        are ['auc', 'ks']. For multi-classification tasks, default metrics are ['accuracy', 'precision', 'recall']

    use_first_metric_only: bool
        use only the first metric for early stopping

    complete_secure: int, defualt: 0
        if use complete_secure, when use complete secure, build first 'complete secure' tree using only guest features

    sparse_optimization:
        this parameter is abandoned in FATE-1.7.1

    run_goss: bool
        activate Gradient-based One-Side Sampling, which selects large gradient and small
        gradient samples using top_rate and other_rate.

        top_rate: float, the retain ratio of large gradient data, used when run_goss is True

        other_rate: float, the retain ratio of small gradient data, used when run_goss is True

        cipher_compress_error： This param is now abandoned

        cipher_compress: bool, default is True, use cipher compressing to reduce computation cost and transfer cost

        boosting_strategy：str

            std: standard sbt setting

            mix:  alternate using guest/host features to build trees. For example, the first 'tree_num_per_party' trees
                  use guest features,
                  the second k trees use host features, and so on

            layered: only support 2 party, when running layered mode, first 'host_depth' layer will use host features,
                     and then next 'guest_depth' will only use guest features

        work_mode: str
                   This parameter has the same function as boosting_strategy, but is deprecated

        tree_num_per_party: int, every party will alternate build 'tree_num_per_party' trees until reach max tree num, this
                            param is valid when boosting_strategy is mix

        guest_depth: int, guest will build last guest_depth of a decision tree using guest features, is valid when boosting_strategy
                     is layered

        host_depth: int, host will build first host_depth of a decision tree using host features, is valid when work boosting_strategy
                    layered


        multi_mode: str, decide which mode to use when running multi-classification task:

                    single_output standard gbdt multi-classification strategy

                    multi_output every leaf give a multi-dimension predict, using multi_mode can save time
                                 by learning a model with less trees.

        EINI_inference: bool
            default is False, this option changes the inference algorithm used in predict tasks.
            a secure prediction method that hides decision path to enhance security in the inference
            step. This method is insprired by EINI inference algorithm.

        EINI_random_mask: bool
            default is False
            multiply predict result by a random float number to confuse original predict result. This operation further
            enhances the security of naive EINI algorithm.

        EINI_complexity_check: bool
            default is False
            check the complexity of tree models when running EINI algorithms. Complexity models are easy to hide their
            decision path, while simple tree models are not, therefore if a tree model is too simple, it is not allowed
            to run EINI predict algorithms.

    """

    def __init__(self, tree_param: DecisionTreeParam = DecisionTreeParam(), task_type=consts.CLASSIFICATION,
                 objective_param=ObjectiveParam(),
                 learning_rate=0.3, num_trees=5, subsample_feature_rate=1.0, n_iter_no_change=True,
                 tol=0.0001, encrypt_param=EncryptParam(),
                 bin_num=32,
                 encrypted_mode_calculator_param=EncryptedModeCalculatorParam(),
                 predict_param=PredictParam(), cv_param=CrossValidationParam(),
                 validation_freqs=None, early_stopping_rounds=None, use_missing=False, zero_as_missing=False,
                 complete_secure=False, metrics=None, use_first_metric_only=False, random_seed=100,
                 binning_error=consts.DEFAULT_RELATIVE_ERROR,
                 sparse_optimization=False, run_goss=False, top_rate=0.2, other_rate=0.1,
                 cipher_compress_error=None, cipher_compress=0, new_ver=True, boosting_strategy=consts.STD_TREE,
                 work_mode=None, tree_num_per_party=1, guest_depth=2, host_depth=3, callback_param=CallbackParam(),
                 multi_mode=consts.SINGLE_OUTPUT, EINI_inference=False, EINI_random_mask=False,
                 EINI_complexity_check=False):

        super(HeteroSecureBoostParam, self).__init__(task_type, objective_param, learning_rate, num_trees,
                                                     subsample_feature_rate, n_iter_no_change, tol, encrypt_param,
                                                     bin_num, encrypted_mode_calculator_param, predict_param, cv_param,
                                                     validation_freqs, early_stopping_rounds, metrics=metrics,
                                                     use_first_metric_only=use_first_metric_only,
                                                     random_seed=random_seed,
                                                     binning_error=binning_error)

        self.tree_param = copy.deepcopy(tree_param)
        self.zero_as_missing = zero_as_missing
        self.use_missing = use_missing
        self.complete_secure = complete_secure
        self.sparse_optimization = sparse_optimization
        self.run_goss = run_goss
        self.top_rate = top_rate
        self.other_rate = other_rate
        self.cipher_compress_error = cipher_compress_error
        self.cipher_compress = cipher_compress
        self.new_ver = new_ver
        self.EINI_inference = EINI_inference
        self.EINI_random_mask = EINI_random_mask
        self.EINI_complexity_check = EINI_complexity_check
        self.boosting_strategy = boosting_strategy
        self.work_mode = work_mode
        self.tree_num_per_party = tree_num_per_party
        self.guest_depth = guest_depth
        self.host_depth = host_depth
        self.callback_param = copy.deepcopy(callback_param)
        self.multi_mode = multi_mode

    def check(self):

        super(HeteroSecureBoostParam, self).check()
        self.tree_param.check()
        if not isinstance(self.use_missing, bool):
            raise ValueError('use missing should be bool type')
        if not isinstance(self.zero_as_missing, bool):
            raise ValueError('zero as missing should be bool type')
        self.check_boolean(self.run_goss, 'run goss')
        self.check_decimal_float(self.top_rate, 'top rate')
        self.check_decimal_float(self.other_rate, 'other rate')
        self.check_positive_number(self.other_rate, 'other_rate')
        self.check_positive_number(self.top_rate, 'top_rate')
        self.check_boolean(self.new_ver, 'code version switcher')
        self.check_boolean(self.cipher_compress, 'cipher compress')
        self.check_boolean(self.EINI_inference, 'eini inference')
        self.check_boolean(self.EINI_random_mask, 'eini random mask')
        self.check_boolean(self.EINI_complexity_check, 'eini complexity check')

        assert isinstance(self.complete_secure,
                          int) and self.complete_secure >= 0, "complete secure should be an int >= 0"

        if self.work_mode is not None:
            self.boosting_strategy = self.work_mode

        if self.multi_mode not in [consts.SINGLE_OUTPUT, consts.MULTI_OUTPUT]:
            raise ValueError('unsupported multi-classification mode')
        if self.multi_mode == consts.MULTI_OUTPUT:
            if self.boosting_strategy != consts.STD_TREE:
                raise ValueError('MO trees only works when boosting strategy is std tree')
            if not self.cipher_compress:
                raise ValueError('Mo trees only works when cipher compress is enabled')

        if self.boosting_strategy not in [consts.STD_TREE, consts.LAYERED_TREE, consts.MIX_TREE]:
            raise ValueError('unknown sbt boosting strategy{}'.format(self.boosting_strategy))

        for p in ["early_stopping_rounds", "validation_freqs", "metrics",
                  "use_first_metric_only"]:
            # if self._warn_to_deprecate_param(p, "", ""):
            if self._deprecated_params_set.get(p):
                if "callback_param" in self.get_user_feeded():
                    raise ValueError(f"{p} and callback param should not be set simultaneously，"
                                     f"{self._deprecated_params_set}, {self.get_user_feeded()}")
                else:
                    self.callback_param.callbacks = ["PerformanceEvaluate"]
                break

        descr = "boosting_param's"

        if self._warn_to_deprecate_param("validation_freqs", descr, "callback_param's 'validation_freqs'"):
            self.callback_param.validation_freqs = self.validation_freqs

        if self._warn_to_deprecate_param("early_stopping_rounds", descr, "callback_param's 'early_stopping_rounds'"):
            self.callback_param.early_stopping_rounds = self.early_stopping_rounds

        if self._warn_to_deprecate_param("metrics", descr, "callback_param's 'metrics'"):
            self.callback_param.metrics = self.metrics

        if self._warn_to_deprecate_param("use_first_metric_only", descr, "callback_param's 'use_first_metric_only'"):
            self.callback_param.use_first_metric_only = self.use_first_metric_only

        if self.top_rate + self.other_rate >= 1:
            raise ValueError('sum of top rate and other rate should be smaller than 1')

        return True


class HomoSecureBoostParam(BoostingParam):
    """
    Parameters
    ----------
    backend: {'distributed', 'memory'}
        decides which backend to use when computing histograms for homo-sbt
    """

    def __init__(self, tree_param: DecisionTreeParam = DecisionTreeParam(), task_type=consts.CLASSIFICATION,
                 objective_param=ObjectiveParam(),
                 learning_rate=0.3, num_trees=5, subsample_feature_rate=1, n_iter_no_change=True,
                 tol=0.0001, bin_num=32, predict_param=PredictParam(), cv_param=CrossValidationParam(),
                 validation_freqs=None, use_missing=False, zero_as_missing=False, random_seed=100,
                 binning_error=consts.DEFAULT_RELATIVE_ERROR, backend=consts.DISTRIBUTED_BACKEND,
                 callback_param=CallbackParam(), multi_mode=consts.SINGLE_OUTPUT):

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
                                                   random_seed=random_seed,
                                                   binning_error=binning_error
                                                   )
        self.use_missing = use_missing
        self.zero_as_missing = zero_as_missing
        self.tree_param = copy.deepcopy(tree_param)
        self.backend = backend
        self.callback_param = copy.deepcopy(callback_param)
        self.multi_mode = multi_mode

    def check(self):

        super(HomoSecureBoostParam, self).check()
        self.tree_param.check()
        if not isinstance(self.use_missing, bool):
            raise ValueError('use missing should be bool type')
        if not isinstance(self.zero_as_missing, bool):
            raise ValueError('zero as missing should be bool type')
        if self.backend not in [consts.MEMORY_BACKEND, consts.DISTRIBUTED_BACKEND]:
            raise ValueError('unsupported backend')
        if self.multi_mode not in [consts.SINGLE_OUTPUT, consts.MULTI_OUTPUT]:
            raise ValueError('unsupported multi-classification mode')

        for p in ["validation_freqs", "metrics"]:
            # if self._warn_to_deprecate_param(p, "", ""):
            if self._deprecated_params_set.get(p):
                if "callback_param" in self.get_user_feeded():
                    raise ValueError(f"{p} and callback param should not be set simultaneously，"
                                     f"{self._deprecated_params_set}, {self.get_user_feeded()}")
                else:
                    self.callback_param.callbacks = ["PerformanceEvaluate"]
                break

        descr = "boosting_param's"

        if self._warn_to_deprecate_param("validation_freqs", descr, "callback_param's 'validation_freqs'"):
            self.callback_param.validation_freqs = self.validation_freqs

        if self._warn_to_deprecate_param("metrics", descr, "callback_param's 'metrics'"):
            self.callback_param.metrics = self.metrics

        if self.multi_mode not in [consts.SINGLE_OUTPUT, consts.MULTI_OUTPUT]:
            raise ValueError('unsupported multi-classification mode')

        if self.multi_mode == consts.MULTI_OUTPUT:
            if self.task_type == consts.REGRESSION:
                raise ValueError('regression tasks not support multi-output trees')

        return True
