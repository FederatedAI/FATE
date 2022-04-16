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

import copy
from federatedml.param.intersect_param import IntersectParam
from types import SimpleNamespace
from federatedml.param.base_param import BaseParam, deprecated_param
from federatedml.util import consts
from federatedml.param.encrypt_param import EncryptParam
from federatedml.param.encrypted_mode_calculation_param import EncryptedModeCalculatorParam
from federatedml.param.predict_param import PredictParam
from federatedml.param.callback_param import CallbackParam

deprecated_param_list = ["validation_freqs", "metrics"]


@deprecated_param(*deprecated_param_list)
class FTLParam(BaseParam):

    def __init__(self, alpha=1, tol=0.000001,
                 n_iter_no_change=False, validation_freqs=None, optimizer={'optimizer': 'Adam', 'learning_rate': 0.01},
                 nn_define={}, epochs=1, intersect_param=IntersectParam(consts.RSA), config_type='keras', batch_size=-1,
                 encrypte_param=EncryptParam(),
                 encrypted_mode_calculator_param=EncryptedModeCalculatorParam(mode="confusion_opt"),
                 predict_param=PredictParam(), mode='plain', communication_efficient=False,
                 local_round=5, callback_param=CallbackParam()):
        """
        Parameters
        ----------
        alpha : float
            a loss coefficient defined in paper, it defines the importance of alignment loss
        tol : float
            loss tolerance
        n_iter_no_change : bool
            check loss convergence or not
        validation_freqs : None or positive integer or container object in python
            Do validation in training process or Not.
            if equals None, will not do validation in train process;
            if equals positive integer, will validate data every validation_freqs epochs passes;
            if container object in python, will validate data if epochs belong to this container.
            e.g. validation_freqs = [10, 15], will validate data when epoch equals to 10 and 15.
            The default value is None, 1 is suggested. You can set it to a number larger than 1 in order to
            speed up training by skipping validation rounds. When it is larger than 1, a number which is
            divisible by "epochs" is recommended, otherwise, you will miss the validation scores
            of last training epoch.
        optimizer : str or dict
            optimizer method, accept following types:
            1. a string, one of "Adadelta", "Adagrad", "Adam", "Adamax", "Nadam", "RMSprop", "SGD"
            2. a dict, with a required key-value pair keyed by "optimizer",
                with optional key-value pairs such as learning rate.
            defaults to "SGD"
        nn_define : dict
            a dict represents the structure of neural network, it can be output by tf-keras
        epochs : int
            epochs num
        intersect_param
            define the intersect method
        config_type : {'tf-keras'}
            config type
        batch_size : int
            batch size when computing transformed feature embedding, -1 use full data.
        encrypte_param
            encrypted param
        encrypted_mode_calculator_param
            encrypted mode calculator param:
        predict_param
            predict param
        mode: {"plain", "encrypted"}
            plain: will not use any encrypt algorithms, data exchanged in plaintext
            encrypted: use paillier to encrypt gradients
        communication_efficient: bool
            will use communication efficient or not. when communication efficient is enabled, FTL model will
            update gradients by several local rounds using intermediate data
        local_round: int
            local update round when using communication efficient
        """

        super(FTLParam, self).__init__()
        self.alpha = alpha
        self.tol = tol
        self.n_iter_no_change = n_iter_no_change
        self.validation_freqs = validation_freqs
        self.optimizer = optimizer
        self.nn_define = nn_define
        self.epochs = epochs
        self.intersect_param = copy.deepcopy(intersect_param)
        self.config_type = config_type
        self.batch_size = batch_size
        self.encrypted_mode_calculator_param = copy.deepcopy(encrypted_mode_calculator_param)
        self.encrypt_param = copy.deepcopy(encrypte_param)
        self.predict_param = copy.deepcopy(predict_param)
        self.mode = mode
        self.communication_efficient = communication_efficient
        self.local_round = local_round
        self.callback_param = copy.deepcopy(callback_param)

    def check(self):
        self.intersect_param.check()
        self.encrypt_param.check()
        self.encrypted_mode_calculator_param.check()

        self.optimizer = self._parse_optimizer(self.optimizer)

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

        for p in deprecated_param_list:
            # if self._warn_to_deprecate_param(p, "", ""):
            if self._deprecated_params_set.get(p):
                if "callback_param" in self.get_user_feeded():
                    raise ValueError(f"{p} and callback param should not be set simultaneously，"
                                     f"{self._deprecated_params_set}, {self.get_user_feeded()}")
                else:
                    self.callback_param.callbacks = ["PerformanceEvaluate"]
                break

        descr = "ftl's"

        if self._warn_to_deprecate_param("validation_freqs", descr, "callback_param's 'validation_freqs'"):
            self.callback_param.validation_freqs = self.validation_freqs

        if self._warn_to_deprecate_param("metrics", descr, "callback_param's 'metrics'"):
            self.callback_param.metrics = self.metrics

        if self.validation_freqs is None:
            pass
        elif isinstance(self.validation_freqs, int):
            if self.validation_freqs < 1:
                raise ValueError("validation_freqs should be larger than 0 when it's integer")
        elif not isinstance(self.validation_freqs, collections.Container):
            raise ValueError("validation_freqs should be None or positive integer or container")

        assert isinstance(self.communication_efficient, bool), 'communication efficient must be a boolean'
        assert self.mode in [
            'encrypted', 'plain'], 'mode options: encrpyted or plain, but {} is offered'.format(
            self.mode)

        self.check_positive_integer(self.epochs, 'epochs')
        self.check_positive_number(self.alpha, 'alpha')
        self.check_positive_integer(self.local_round, 'local round')

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
