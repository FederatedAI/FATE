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

from numpy import random

from federatedml.param.base_param import BaseParam
from federatedml.param.cross_validation_param import CrossValidationParam
from federatedml.param.encrypt_param import EncryptParam
from federatedml.param.predict_param import PredictParam
from federatedml.util import consts


class RandomParam(BaseParam):
    def __init__(self, method="normal", loc=0, scale=1.0, seed=None):
        self.method = method
        self.loc = loc
        self.scale = scale
        self.seed = seed

    def check(self):
        try:
            func = getattr(random, self.method)()
        except AttributeError:
            raise ValueError("method not supported".format(self.method))

        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed should be None or integer")

        if not isinstance(self.loc, (int, float)):
            raise ValueError("loc should be numeric")

        if not isinstance(self.scale, (int, float)):
            raise ValueError("scale should be numeric")


class HeteroNNParam(BaseParam):
    def __init__(self,
                 task_type='binary',
                 config_type="nn",
                 bottom_nn_define=None,
                 top_nn_define=None,
                 interactive_layer_define=None,
                 interactive_layer_lr=0.9,
                 optimizer='SGD',
                 loss=None,
                 metrics=None,
                 epochs=100,
                 batch_size=-1,
                 early_stop="diff",
                 tol=1e-5,
                 encrypt_param=EncryptParam(),
                 predict_param=PredictParam(),
                 random_param=RandomParam(),
                 cv_param=CrossValidationParam()):
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
        self.metrics = metrics
        self.optimizer = optimizer
        self.loss = loss

        self.encrypt_param = copy.deepcopy(encrypt_param)
        self.predict_param = copy.deepcopy(predict_param)
        self.random_param = copy.deepcopy(random_param)
        self.cv_param = copy.deepcopy(cv_param)

    def check(self):
        self.optimizer = self._parse_optimizer(self.optimizer)
        supported_config_type = ["keras"]
        if self.config_type not in supported_config_type:
            raise ValueError(f"config_type should be one of {supported_config_type}")

        if not isinstance(self.tol, (int, float)):
            raise ValueError("tol should be numeric")

        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("epochs should be a positive integer")

        if not self.bottom_nn_define or not isinstance(self.bottom_nn_define, dict):
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
            raise  ValueError("early stop should be diff in this version")

        self.predict_param.check()
        self.random_param.check()

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
