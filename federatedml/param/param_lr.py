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

import copy

from federatedml.util import consts


class EncryptedModeCalculatorParam(object):
    """
    Define the encrypted_mode_calulator parameters.

    Parameters
    ----------
    mode: str, support 'strict', 'fast', 'balance' only, default: strict

    re_encrypted_rate: float or int, numeric number, use when mode equals to 'strict', defualt: 1

    """

    def __init__(self, mode="strict", re_encrypted_rate=1):
        self.mode = mode
        self.re_encrypted_rate = re_encrypted_rate


class EncryptParam(object):
    """
    Define encryption method that used in federated ml.

    Parameters
    ----------
    method : str, default: 'Paillier'
        If method is 'Paillier', Paillier encryption will be used for federated ml.
        To use non-encryption version in HomoLR, just set this parameter to be any other str.
        For detail of Paillier encryption, please check out the paper mentioned in README file.

    key_length : int, default: 1024
        Used to specify the length of key in this encryption method. Only needed when method is 'Paillier'

    """

    def __init__(self, method=consts.PAILLIER, key_length=1024):
        self.method = method
        self.key_length = key_length


class InitParam(object):
    """
    Initialize Parameters used in initializing a model.

    Parameters
    ----------
    init_method : str, 'random_uniform', 'random_normal', 'ones', 'zeros' or 'const'. default: 'random_uniform'
        Initial method.

    init_const : int or float, default: 1
        Required when init_method is 'const'. Specify the constant.

    fit_intercept : bool, default: True
        Whether to initialize the intercept or not.

    """

    def __init__(self, init_method='random_uniform', init_const=1, fit_intercept=True):
        self.init_method = init_method
        self.init_const = init_const
        self.fit_intercept = fit_intercept


class PredictParam(object):
    """
    Define the predict method of HomoLR, HeteroLR, SecureBoosting

    Parameters
    ----------
    with_proba: bool, Specify whether the result contains probability

    threshold: float or int, The threshold use to separate positive and negative class. Normally, it should be (0,1)
    """

    def __init__(self, with_proba=True, threshold=0.5):
        self.with_proba = with_proba
        self.threshold = threshold


class EvaluateParam(object):
    """
    Define the evaluation method of binary/multiple classification and regression

    Parameters
    ----------
    metrics: A list of evaluate index. Support 'auc', 'ks', 'lift', 'precision' ,'recall' and 'accuracy', 'explain_variance',
            'mean_absolute_error', 'mean_squared_error', 'mean_squared_log_error','median_absolute_error','r2_score','root_mean_squared_error'.
            For example, metrics can be set as ['auc', 'precision', 'recall'], then the results of these indexes will be output.

    classi_type: string, support 'binary' for HomoLR, HeteroLR and Secureboosting. support 'regression' for Secureboosting. 'multi' is not support these version

    pos_label: specify positive label type, can be int, float and str, this depend on the data's label, this parameter effective only for 'binary'

    thresholds: A list of threshold. Specify the threshold use to separate positive and negative class. for example [0.1, 0.3,0.5], this parameter effective only for 'binary'
    """

    def __init__(self, metrics=None, classi_type="binary", pos_label=1, thresholds=None):
        if metrics is None:
            metrics = []
        self.metrics = metrics
        self.classi_type = classi_type
        self.pos_label = pos_label
        if thresholds is None:
            thresholds = [0.5]

        self.thresholds = thresholds


class CrossValidationParam(object):
    """
    Define cross validation params

    Parameters
    ----------
    n_splits: int, default: 5
        Specify how many splits used in KFold

    mode: str, default: 'Hetero'
        Indicate what mode is current task

    role: str, default: 'Guest'
        Indicate what role is current party

    shuffle: bool, default: True
        Define whether do shuffle before KFold or not.

    random_seed: int, default: 1
        Specify the random seed for numpy shuffle

    """

    def __init__(self, n_splits=5, shuffle=True, random_seed=1,
                 evaluate_param=EvaluateParam(), need_cv=False):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.evaluate_param = copy.deepcopy(evaluate_param)
        self.need_cv = need_cv


class LogisticParam(object):
    """
    Parameters used for Logistic Regression both for Homo mode or Hetero mode.

    Parameters
    ----------
    penalty : str, 'L1' or 'L2'. default: 'L2'
        Penalty method used in LR. Please note that, when using encrypted version in HomoLR,
        'L1' is not supported.

    eps : float, default: 1e-5
        The tolerance of convergence

    alpha : float, default: 1.0
        Regularization strength coefficient.

    optimizer : str, 'sgd', 'rmsprop', 'adam' or 'adagrad', default: 'sgd'
        Optimize method

    party_weight : int or float, default: 1
        Required in Homo LR. Setting the weight of model updated for this party.
        The higher weight set, the higher influence made for this party when updating model.

    batch_size : int, default: -1
        Batch size when updating model. -1 means use all data in a batch. i.e. Not to use mini-batch strategy.

    learning_rate : float, default: 0.01
        Learning rate

    max_iter : int, default: 100
        The maximum iteration for training.

    converge_func : str, 'diff' or 'abs', default: 'diff'
        Method used to judge converge or not.
            a)	diff： Use difference of loss between two iterations to judge whether converge.
            b)	abs: Use the absolute value of loss to judge whether converge. i.e. if loss < eps, it is converged.

    re_encrypt_batches : int, default: 2
        Required when using encrypted version HomoLR. Since multiple batch updating coefficient may cause
        overflow error. The model need to be re-encrypt for every several batches. Please be careful when setting
        this parameter. Too large batches may cause training failure.

    model_path : Abandoned

    table_name : Abandoned

    """

    def __init__(self, penalty='L2',
                 eps=1e-5, alpha=1.0, optimizer='sgd', party_weight=1,
                 batch_size=-1, learning_rate=0.01, init_param=InitParam(),
                 max_iter=100, converge_func='diff',
                 encrypt_param=EncryptParam(), re_encrypt_batches=2,
                 model_path='lr_model', encrypted_mode_calculator_param=EncryptedModeCalculatorParam(),
                 need_run=True, predict_param=PredictParam(), cv_param=CrossValidationParam()):
        self.penalty = penalty
        self.eps = eps
        self.alpha = alpha
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.init_param = copy.deepcopy(init_param)
        self.max_iter = max_iter
        self.converge_func = converge_func
        self.encrypt_param = copy.deepcopy(encrypt_param)
        self.re_encrypt_batches = re_encrypt_batches
        self.model_path = model_path
        self.party_weight = party_weight
        self.encrypted_mode_calculator_param = copy.deepcopy(encrypted_mode_calculator_param)
        self.need_run = need_run
        self.predict_param = copy.deepcopy(predict_param)
        self.cv_param = copy.deepcopy(cv_param)
