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
import collections
from types import SimpleNamespace

from pipeline.param.base_param import BaseParam
from pipeline.param.cross_validation_param import CrossValidationParam
from pipeline.param.encrypt_param import EncryptParam
from pipeline.param.encrypted_mode_calculation_param import EncryptedModeCalculatorParam
from pipeline.param.predict_param import PredictParam
from pipeline.param import consts


class SelectorParam(object):
    def __init__(self, method=None, beta=1, selective_size=consts.SELECTIVE_SIZE, min_prob=0, random_state=None):
        self.method = method
        self.selective_size = selective_size
        self.beta = beta
        self.min_prob = min_prob
        self.random_state = random_state

    def check(self):
        if self.method is not None and self.method not in ["relative"]:
            raise ValueError('selective method should be None be "relative"')


class HeteroNNParam(BaseParam):
    """
    Parameters used for Homo Neural Network.

    Args:
        task_type: str, task type of hetero nn model, one of 'classification', 'regression'.
        config_type: str, accept "keras" only.
        bottom_nn_define: a dict represents the structure of bottom neural network.
        interactive_layer_define: a dict represents the structure of interactive layer.
        interactive_layer_lr: float, the learning rate of interactive layer.
        top_nn_define: a dict represents the structure of top neural network.
        optimizer: optimizer method, accept following types:
            1. a string, one of "Adadelta", "Adagrad", "Adam", "Adamax", "Nadam", "RMSprop", "SGD"
            2. a dict, with a required key-value pair keyed by "optimizer",
                with optional key-value pairs such as learning rate.
            defaults to "SGD"
        loss:  str, a string to define loss function used
        early_stopping_rounds: int, default: None
        Will stop training if one metric doesn’t improve in last early_stopping_round rounds
        metrics: list, default: None
            Indicate when executing evaluation during train process, which metrics will be used. If not set,
            default metrics for specific task type will be used. As for binary classification, default metrics are
            ['auc', 'ks'], for regression tasks, default metrics are ['root_mean_squared_error', 'mean_absolute_error'],
            [ACCURACY, PRECISION, RECALL] for multi-classification task
        use_first_metric_only: bool, default: False
            Indicate whether to use the first metric in `metrics` as the only criterion for early stopping judgement.
        epochs: int, the maximum iteration for aggregation in training.
        batch_size : int, batch size when updating model.
            -1 means use all data in a batch. i.e. Not to use mini-batch strategy.
            defaults to -1.
        early_stop : str, accept 'diff' only in this version, default: 'diff'
            Method used to judge converge or not.
                a)	diff： Use difference of loss between two iterations to judge whether converge.
        validation_freqs: None or positive integer or container object in python. Do validation in training process or Not.
                  if equals None, will not do validation in train process;
                  if equals positive integer, will validate data every validation_freqs epochs passes;
                  if container object in python, will validate data if epochs belong to this container.
                    e.g. validation_freqs = [10, 15], will validate data when epoch equals to 10 and 15.
                  Default: None
                  The default value is None, 1 is suggested. You can set it to a number larger than 1 in order to
                  speed up training by skipping validation rounds. When it is larger than 1, a number which is
                  divisible by "epochs" is recommended, otherwise, you will miss the validation scores
                  of last training epoch.
        floating_point_precision: None or integer, if not None, means use floating_point_precision-bit to speed up calculation,
                                   e.g.: convert an x to round(x * 2**floating_point_precision) during Paillier operation, divide
                                          the result by 2**floating_point_precision in the end.
        drop_out_keep_rate: float, should betweend 0 and 1, if not equals to 1.0, will enabled drop out
    """

    def __init__(self,
                 task_type='classification',
                 config_type="keras",
                 bottom_nn_define=None,
                 top_nn_define=None,
                 interactive_layer_define=None,
                 interactive_layer_lr=0.9,
                 optimizer='SGD',
                 loss=None,
                 epochs=100,
                 batch_size=-1,
                 early_stop="diff",
                 tol=1e-5,
                 encrypt_param=EncryptParam(),
                 encrypted_mode_calculator_param=EncryptedModeCalculatorParam(mode="confusion_opt"),
                 predict_param=PredictParam(),
                 cv_param=CrossValidationParam(),
                 validation_freqs=None,
                 early_stopping_rounds=None,
                 metrics=None,
                 use_first_metric_only=True,
                 selector_param=SelectorParam(),
                 floating_point_precision=23,
                 drop_out_keep_rate=1.0):
        super(HeteroNNParam, self).__init__()

        self.task_type = task_type
        self.config_type = config_type
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

        self.drop_out_keep_rate = drop_out_keep_rate

    def check(self):
        self.optimizer = self._parse_optimizer(self.optimizer)
        supported_config_type = ["keras"]

        if self.task_type not in ["classification", "regression"]:
            raise ValueError("config_type should be classification or regression")

        if self.config_type not in supported_config_type:
            raise ValueError(f"config_type should be one of {supported_config_type}")

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

        if self.validation_freqs is None:
            pass
        elif isinstance(self.validation_freqs, int):
            if self.validation_freqs < 1:
                raise ValueError("validation_freqs should be larger than 0 when it's integer")
        elif not isinstance(self.validation_freqs, collections.Container):
            raise ValueError("validation_freqs should be None or positive integer or container")

        if self.early_stopping_rounds and not isinstance(self.early_stopping_rounds, int):
            raise ValueError("early stopping rounds should be None or int larger than 0")
        if self.early_stopping_rounds and isinstance(self.early_stopping_rounds, int):
            if self.early_stopping_rounds < 1:
                raise ValueError("early stopping should be larger than 0 when it's integer")
            if not self.validation_freqs:
                raise ValueError("If early stopping rounds is setting, validation_freqs should not be null")

        if self.metrics is not None and not isinstance(self.metrics, list):
            raise ValueError("metrics should be a list")

        if not isinstance(self.use_first_metric_only, bool):
            raise ValueError("use_first_metric_only should be a boolean")

        if self.floating_point_precision is not None and \
                (not isinstance(self.floating_point_precision, int) or \
                 self.floating_point_precision < 0 or self.floating_point_precision > 64):
            raise ValueError("floating point precision should be null or a integer between 0 and 64")

        self.encrypt_param.check()
        self.encrypted_model_calculator_param.check()
        self.predict_param.check()

    @staticmethod
    def _parse_optimizer(opt):
        """
        Examples:

            1. "optimize": "SGD"
            2. "optimize": {
                "optimizer": "SGD",
                "learning_rate": 0.05
            }
        """

        kwargs = {}
        if isinstance(opt, str):
            return SimpleNamespace(optimizer=opt, kwargs=kwargs)
        elif isinstance(opt, dict):
            optimizer = opt.get("optimizer", kwargs)
            if not optimizer:
                raise ValueError(f"optimizer config: {opt} invalid")
            kwargs = {k: v for k, v in opt.items() if k != "optimizer"}
            return SimpleNamespace(optimizer=optimizer, kwargs=kwargs)
        else:
            raise ValueError(f"invalid type for optimize: {type(opt)}")
