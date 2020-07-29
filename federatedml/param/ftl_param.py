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
import collections

from federatedml.param.intersect_param import IntersectParam
from types import SimpleNamespace
from federatedml.param.base_param import BaseParam
from federatedml.util import consts
from federatedml.param.encrypt_param import EncryptParam
from federatedml.param.encrypted_mode_calculation_param import EncryptedModeCalculatorParam
from federatedml.param.predict_param import PredictParam


class FTLParam(BaseParam):

    def __init__(self, learning_rate=0.01, alpha=1, tol=0.000001,
                 n_iter_no_change=False, validation_freqs=None, optimizer='Adam', nn_define={}, epochs=1
                 , intersect_param=IntersectParam(consts.RSA), config_type='keras', batch_size=64,
                 encrypte_param=EncryptParam(),
                 encrypted_mode_calculator_param=EncryptedModeCalculatorParam(mode="confusion_opt"),
                 predict_param=PredictParam(), mode='plain', communication_efficient=False,
                 local_round=5,):

        super(FTLParam, self).__init__()
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.validation_freqs = validation_freqs
        self.optimizer = optimizer
        self.nn_define = nn_define
        self.epochs = epochs
        self.intersect_param = intersect_param
        self.config_type = config_type
        self.batch_size = batch_size
        self.encrypted_mode_calculator_param = encrypted_mode_calculator_param
        self.encrypt_param = encrypte_param
        self.predict_param = predict_param
        self.mode = mode
        self.communication_efficient = communication_efficient
        self.local_round = local_round

    def check(self):
        self.intersect_param.check()
        self.encrypt_param.check()
        self.encrypted_mode_calculator_param.check()

        self.optimizer = self._parse_optimizer({'optimizer': self.optimizer, 'learning_rate': self.learning_rate})

        supported_config_type = ["keras"]
        if self.config_type not in supported_config_type:
            raise ValueError(f"config_type should be one of {supported_config_type}")

        if not isinstance(self.tol, (int, float)):
            raise ValueError("tol should be numeric")

        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("epochs should be a positive integer")

        if self.nn_define and not isinstance(self.nn_define, dict):
            raise ValueError("bottom_nn_define should be a dict defining the structure of neural network")

        if self.batch_size != -1:
            if not isinstance(self.batch_size, int) \
                    or self.batch_size < consts.MIN_BATCH_SIZE:
                raise ValueError(
                    " {} not supported, should be larger than 10 or -1 represent for all data".format(self.batch_size))

        if self.validation_freqs is None:
            pass
        elif isinstance(self.validation_freqs, int):
            if self.validation_freqs < 1:
                raise ValueError("validation_freqs should be larger than 0 when it's integer")
        elif not isinstance(self.validation_freqs, collections.Container):
            raise ValueError("validation_freqs should be None or positive integer or container")

        assert type(self.communication_efficient) is bool, 'communication efficient must be a boolean'
        assert self.mode in ['encrypted', 'plain'], 'mode options: encrpyted or plain, but {} is offered'.format(self.mode)
        assert type(self.epochs) == int and self.epochs > 0

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

