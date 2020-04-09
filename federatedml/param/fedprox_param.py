#!/usr/bin/env python
# -*- coding: utf-8 -*-

# JS changed for test and debug purpose
import copy

from federatedml.param.base_param import BaseParam
from federatedml.param.cross_validation_param import CrossValidationParam
from federatedml.param.encrypt_param import EncryptParam
from federatedml.param.encrypted_mode_calculation_param import EncryptedModeCalculatorParam
from federatedml.param.init_model_param import InitParam
from federatedml.param.logistic_regression_param import LogisticParam
from federatedml.param.one_vs_rest_param import OneVsRestParam
from federatedml.param.predict_param import PredictParam
from federatedml.util import consts


class FedProxParam(LogisticParam):
    """
    Parameters
    ----------
    re_encrypt_batches : int, default: 2
        Required when using encrypted version HomoLR. Since multiple batch updating coefficient may cause
        overflow error. The model need to be re-encrypt for every several batches. Please be careful when setting
        this parameter. Too large batches may cause training failure.

    aggregate_iters : int, default: 1
        Indicate how many iterations are aggregated once.

    """
    def __init__(self, penalty='L2',
                 tol=1e-5, alpha=1.0, optimizer='sgd',
                 batch_size=-1, learning_rate=0.01, init_param=InitParam(),
                 max_iter=100, early_stop='diff',
                 encrypt_param=EncryptParam(), re_encrypt_batches=2,
                 predict_param=PredictParam(), cv_param=CrossValidationParam(),
                 decay=1, decay_sqrt=True,
                 aggregate_iters=1, multi_class='ovr', validation_freqs=None
                 ):
        super(FedProxParam, self).__init__(penalty=penalty, tol=tol, alpha=alpha, optimizer=optimizer,
                                                batch_size=batch_size,
                                                learning_rate=learning_rate,
                                                init_param=init_param, max_iter=max_iter, early_stop=early_stop,
                                                encrypt_param=encrypt_param, predict_param=predict_param,
                                                cv_param=cv_param, multi_class=multi_class,
                                                validation_freqs=validation_freqs,
                                                decay=decay, decay_sqrt=decay_sqrt)
        self.re_encrypt_batches = re_encrypt_batches
        self.aggregate_iters = aggregate_iters
        self.mu = 0.1
        self.use_fedprox = True

    def check(self):
        super().check()
        if type(self.re_encrypt_batches).__name__ != "int":
            raise ValueError(
                "AIFELParam's re_encrypt_batches {} not supported, should be int type".format(
                    self.re_encrypt_batches))
        elif self.re_encrypt_batches < 0:
            raise ValueError(
                "AIFELParam's re_encrypt_batches must be greater or equal to 0")

        if not isinstance(self.aggregate_iters, int):
            raise ValueError(
                "AIFELParam's aggregate_iters {} not supported, should be int type".format(
                    self.aggregate_iters))

        if self.encrypt_param.method == consts.PAILLIER:
            if self.optimizer != 'sgd':
                raise ValueError("AIFELParam Paillier encryption mode supports 'sgd' optimizer method only.")

            if self.penalty == consts.L1_PENALTY:
                raise ValueError("AIFELParam Paillier encryption mode supports 'L2' penalty or None only.")
        return True
