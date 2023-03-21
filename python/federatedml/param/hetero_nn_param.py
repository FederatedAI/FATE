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
import copy
from types import SimpleNamespace

from federatedml.param.base_param import BaseParam
from federatedml.param.base_param import deprecated_param
from federatedml.param.callback_param import CallbackParam
from federatedml.param.cross_validation_param import CrossValidationParam
from federatedml.param.encrypt_param import EncryptParam
from federatedml.param.encrypted_mode_calculation_param import EncryptedModeCalculatorParam
from federatedml.param.predict_param import PredictParam
from federatedml.util import consts


class DatasetParam(BaseParam):

    def __init__(self, dataset_name=None, **kwargs):
        super(DatasetParam, self).__init__()
        self.dataset_name = dataset_name
        self.param = kwargs

    def check(self):
        if self.dataset_name is not None:
            self.check_string(self.dataset_name, 'dataset_name')

    def to_dict(self):
        ret = {'dataset_name': self.dataset_name, 'param': self.param}
        return ret


class SelectorParam(object):
    """
    Parameters
    ----------
    method: None or str
        back propagation select method, accept "relative" only, default: None
    selective_size: int
        deque size to use, store the most recent selective_size historical loss, default: 1024
    beta: int
        sample whose selective probability >= power(np.random, beta) will be selected
    min_prob: Numeric
        selective probability is max(min_prob, rank_rate)
    """

    def __init__(self, method=None, beta=1, selective_size=consts.SELECTIVE_SIZE, min_prob=0, random_state=None):
        self.method = method
        self.selective_size = selective_size
        self.beta = beta
        self.min_prob = min_prob
        self.random_state = random_state

    def check(self):
        if self.method is not None and self.method not in ["relative"]:
            raise ValueError('selective method should be None be "relative"')

        if not isinstance(self.selective_size, int) or self.selective_size <= 0:
            raise ValueError("selective size should be a positive integer")

        if not isinstance(self.beta, int):
            raise ValueError("beta should be integer")

        if not isinstance(self.min_prob, (float, int)):
            raise ValueError("min_prob should be numeric")


class CoAEConfuserParam(BaseParam):
    """
    A label protect mechanism proposed in paper: "Batch Label Inference and Replacement Attacks in Black-Boxed Vertical Federated Learning"
    paper link: https://arxiv.org/abs/2112.05409
    Convert true labels to fake soft labels by using an auto-encoder.

    Args:
        enable: boolean
            run CoAE or not
        epoch: None or int
            auto-encoder training epochs
        lr: float
            auto-encoder learning rate
        lambda1: float
            parameter to control the difference between true labels and fake soft labels. Larger the parameter,
            autoencoder will give more attention to making true labels and fake soft label different.
        lambda2: float
            parameter to control entropy loss, see original paper for details
        verbose: boolean
            print loss log while training auto encoder
    """

    def __init__(self, enable=False, epoch=50, lr=0.001, lambda1=1.0, lambda2=2.0, verbose=False):
        super(CoAEConfuserParam, self).__init__()
        self.enable = enable
        self.epoch = epoch
        self.lr = lr
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.verbose = verbose

    def check(self):

        self.check_boolean(self.enable, 'enable')

        if not isinstance(self.epoch, int) or self.epoch <= 0:
            raise ValueError("epoch should be a positive integer")

        if not isinstance(self.lr, float):
            raise ValueError('lr should be a float number')

        if not isinstance(self.lambda1, float):
            raise ValueError('lambda1 should be a float number')

        if not isinstance(self.lambda2, float):
            raise ValueError('lambda2 should be a float number')

        self.check_boolean(self.verbose, 'verbose')


@deprecated_param("validation_freqs", "early_stopping_rounds", "metrics", "use_first_metric_only")
class HeteroNNParam(BaseParam):
    """
    Parameters used for Hetero Neural Network.

    Parameters
    ----------
    task_type: str, task type of hetero nn model, one of 'classification', 'regression'.
    bottom_nn_define: a dict represents the structure of bottom neural network.
    interactive_layer_define: a dict represents the structure of interactive layer.
    interactive_layer_lr: float, the learning rate of interactive layer.
    top_nn_define: a dict represents the structure of top neural network.
    optimizer: optimizer method, accept following types:
        1. a string, one of "Adadelta", "Adagrad", "Adam", "Adamax", "Nadam", "RMSprop", "SGD"
        2. a dict, with a required key-value pair keyed by "optimizer",
            with optional key-value pairs such as learning rate.
        defaults to "SGD".
    loss:  str, a string to define loss function used
    epochs: int, the maximum iteration for aggregation in training.
    batch_size : int, batch size when updating model.
        -1 means use all data in a batch. i.e. Not to use mini-batch strategy.
        defaults to -1.
    early_stop : str, accept 'diff' only in this version, default: 'diff'
        Method used to judge converge or not.
            a)	diff： Use difference of loss between two iterations to judge whether converge.
    floating_point_precision: None or integer, if not None, means use floating_point_precision-bit to speed up calculation,
                                e.g.: convert an x to round(x * 2**floating_point_precision) during Paillier operation, divide
                                        the result by 2**floating_point_precision in the end.
    callback_param: CallbackParam object
    """

    def __init__(self,
                 task_type='classification',
                 bottom_nn_define=None,
                 top_nn_define=None,
                 interactive_layer_define=None,
                 interactive_layer_lr=0.9,
                 config_type='pytorch',
                 optimizer='SGD',
                 loss=None,
                 epochs=100,
                 batch_size=-1,
                 early_stop="diff",
                 tol=1e-5,
                 seed=100,
                 encrypt_param=EncryptParam(),
                 encrypted_mode_calculator_param=EncryptedModeCalculatorParam(),
                 predict_param=PredictParam(),
                 cv_param=CrossValidationParam(),
                 validation_freqs=None,
                 early_stopping_rounds=None,
                 metrics=None,
                 use_first_metric_only=True,
                 selector_param=SelectorParam(),
                 floating_point_precision=23,
                 callback_param=CallbackParam(),
                 coae_param=CoAEConfuserParam(),
                 dataset=DatasetParam()
                 ):

        super(HeteroNNParam, self).__init__()

        self.task_type = task_type
        self.bottom_nn_define = bottom_nn_define
        self.interactive_layer_define = interactive_layer_define
        self.interactive_layer_lr = interactive_layer_lr
        self.top_nn_define = top_nn_define
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop = early_stop
        self.tol = tol
        self.optimizer = optimizer
        self.loss = loss
        self.validation_freqs = validation_freqs
        self.early_stopping_rounds = early_stopping_rounds
        self.metrics = metrics or []
        self.use_first_metric_only = use_first_metric_only
        self.encrypt_param = copy.deepcopy(encrypt_param)
        self.encrypted_model_calculator_param = encrypted_mode_calculator_param
        self.predict_param = copy.deepcopy(predict_param)
        self.cv_param = copy.deepcopy(cv_param)
        self.selector_param = selector_param
        self.floating_point_precision = floating_point_precision
        self.callback_param = copy.deepcopy(callback_param)
        self.coae_param = coae_param
        self.dataset = dataset
        self.seed = seed
        self.config_type = 'pytorch'  # pytorch only

    def check(self):

        assert isinstance(self.dataset, DatasetParam), 'dataset must be a DatasetParam()'

        self.dataset.check()

        self.check_positive_integer(self.seed, 'seed')

        if self.task_type not in ["classification", "regression"]:
            raise ValueError("config_type should be classification or regression")

        if not isinstance(self.tol, (int, float)):
            raise ValueError("tol should be numeric")

        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("epochs should be a positive integer")

        if self.bottom_nn_define and not isinstance(self.bottom_nn_define, dict):
            raise ValueError("bottom_nn_define should be a dict defining the structure of neural network")

        if self.top_nn_define and not isinstance(self.top_nn_define, dict):
            raise ValueError("top_nn_define should be a dict defining the structure of neural network")

        if self.interactive_layer_define is not None and not isinstance(self.interactive_layer_define, dict):
            raise ValueError(
                "the interactive_layer_define should be a dict defining the structure of interactive layer")

        if self.batch_size != -1:
            if not isinstance(self.batch_size, int) \
                    or self.batch_size < consts.MIN_BATCH_SIZE:
                raise ValueError(
                    " {} not supported, should be larger than 10 or -1 represent for all data".format(self.batch_size))

        if self.early_stop != "diff":
            raise ValueError("early stop should be diff in this version")

        if self.metrics is not None and not isinstance(self.metrics, list):
            raise ValueError("metrics should be a list")

        if self.floating_point_precision is not None and \
                (not isinstance(self.floating_point_precision, int) or
                 self.floating_point_precision < 0 or self.floating_point_precision > 63):
            raise ValueError("floating point precision should be null or a integer between 0 and 63")

        self.encrypt_param.check()
        self.encrypted_model_calculator_param.check()
        self.predict_param.check()
        self.selector_param.check()
        self.coae_param.check()

        descr = "hetero nn param's "

        for p in ["early_stopping_rounds", "validation_freqs",
                  "use_first_metric_only"]:
            if self._deprecated_params_set.get(p):
                if "callback_param" in self.get_user_feeded():
                    raise ValueError(f"{p} and callback param should not be set simultaneously，"
                                     f"{self._deprecated_params_set}, {self.get_user_feeded()}")
                else:
                    self.callback_param.callbacks = ["PerformanceEvaluate"]
                break

        if self._warn_to_deprecate_param("validation_freqs", descr, "callback_param's 'validation_freqs'"):
            self.callback_param.validation_freqs = self.validation_freqs

        if self._warn_to_deprecate_param("early_stopping_rounds", descr, "callback_param's 'early_stopping_rounds'"):
            self.callback_param.early_stopping_rounds = self.early_stopping_rounds

        if self._warn_to_deprecate_param("metrics", descr, "callback_param's 'metrics'"):
            if self.metrics:
                self.callback_param.metrics = self.metrics

        if self._warn_to_deprecate_param("use_first_metric_only", descr, "callback_param's 'use_first_metric_only'"):
            self.callback_param.use_first_metric_only = self.use_first_metric_only
