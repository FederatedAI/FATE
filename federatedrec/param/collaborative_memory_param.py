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
import json
import typing
from types import SimpleNamespace

from federatedml.param.base_param import BaseParam
from federatedml.param.cross_validation_param import CrossValidationParam
from federatedml.param.init_model_param import InitParam
from federatedml.param.predict_param import PredictParam
from federatedml.util import consts
from federatedml.protobuf.generated import cmn_model_meta_pb2


class CMNInitParam(InitParam):
    def __init__(self, embed_dim=10, init_method='random_normal'):
        super(CMNInitParam, self).__init__()
        self.embed_dim = embed_dim
        self.init_method = init_method

    def check(self):
        if type(self.embed_dim).__name__ not in ["int"] or self.embed_dim < 0:
            raise ValueError(
                "CMNInitParam's embed_dim {} not supported, should be 'int'"
                "and greater than 0".format(
                    self.embed_dim))
        return True


class CMNParam(BaseParam):
    """
    Parameters
    ----------
    optimizer : dict, support optimizers in Keras such as  'SGD', 'RMSprop', 'Adam',  or 'Adagrad',
        default: 'SGD' with learning rate 0.01
        Optimize method

    batch_size : int, default: -1
        Batch size when updating model. -1 means use all data in a batch. i.e. Not to use mini-batch strategy.

    init_param: InitParam object, default: default InitParam object
        Init param method object.

    max_iter : int, default: 100
        The maximum iteration for training.

    early_stop : dict. early_stop includes 'diff', 'weight_diff' and 'abs',
        default: {'early_stop':'diff', 'eps': 1e-5}
        Method used to judge converge or not.
            a)	diff： Use difference of loss between two iterations to judge whether converge.
            b)  weight_diff: Use difference between weights of two consecutive iterations
            c)	abs: Use the absolute value of loss to judge whether converge. i.e. if loss < eps, it is converged.


    predict_param: PredictParam object, default: default PredictParam object

    cv_param: CrossValidationParam object, default: default CrossValidationParam object

    hops: hops count of CMN model

    max_len: max length of neighbors

    l2_coef: l2 penalty coefficient
    """

    def __init__(self,
                 secure_aggregate: bool = True,
                 aggregate_every_n_epoch: int = 1,
                 early_stop: typing.Union[str, dict, SimpleNamespace] = {"early_stop": "diff"},
                 optimizer: typing.Union[str, dict, SimpleNamespace] = {"optimizer": "SGD", "learning_rate": 0.01},
                 batch_size=-1,
                 init_param=CMNInitParam(),
                 max_iter=100,
                 predict_param=PredictParam(),
                 cv_param=CrossValidationParam(),
                 validation_freqs=None,
                 metrics: typing.Union[str, list] = None,
                 loss: str = 'mse',
                 hops=2, neg_count=4, l2_coef=0.1, max_len=20
                 ):
        super(CMNParam, self).__init__()
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.init_param = copy.deepcopy(init_param)
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.predict_param = copy.deepcopy(predict_param)
        self.cv_param = copy.deepcopy(cv_param)
        self.validation_freqs = validation_freqs
        self.hops = hops
        self.max_len = max_len
        self.neg_count = neg_count
        self.l2_coef = l2_coef
        self.secure_aggregate = secure_aggregate
        self.aggregate_every_n_epoch = aggregate_every_n_epoch
        self.loss = loss
        self.metrics = metrics

    def check(self):
        descr = "CMN(collaborative Memory Networks)'s"
        self.early_stop = self._parse_early_stop(self.early_stop)
        self.optimizer = self._parse_optimizer(self.optimizer)
        self.metrics = self._parse_metrics(self.metrics)

        if self.batch_size != -1:
            if type(self.batch_size).__name__ not in ["int"] \
                    or self.batch_size < consts.MIN_BATCH_SIZE:
                raise ValueError(descr + " {} not supported, should be larger than 10 or "
                                         "-1 represent for all data".format(self.batch_size))

        if 'learning_rate' in self.optimizer.__dict__["kwargs"] and \
                (type(self.optimizer.kwargs['learning_rate']).__name__ != "float" or (
                isinstance(self.optimizer.kwargs['learning_rate'], float)) and
                 self.optimizer.kwargs['learning_rate'] < 0):
            raise ValueError(
                "CMN(collaborative Memory Networks)'s optimizer['learning_rate'] {} not supported"
                ", should be float type and greater than 0".format(self.optimizer.kwargs['learning_rate']))

        if 'decay' in self.optimizer.__dict__["kwargs"] and \
                (type(self.optimizer.kwargs['decay']).__name__ != "float" or (
                isinstance(self.optimizer.kwargs['decay'], float)) and
                 self.optimizer.kwargs['decay'] < 0):
            raise ValueError(
                "CMN(collaborative Memory Networks)'s optimizer['decay'] {} not supported, "
                "should be float type and greater than 0".format(self.optimizer.kwargs['decay']))

        self.init_param.check()

        if type(self.max_iter).__name__ != "int" or (isinstance(self.max_iter, int) and self.max_iter < 1):
            raise ValueError(
                "CMN(collaborative Memory Networks)'s max_iter {} not supported, "
                "should be int type and greater than 0".format(
                    self.max_iter))
        elif self.max_iter <= 0:
            raise ValueError(
                "CMN(collaborative Memory Networks)'s max_iter must be greater than or equal to 1")

        if type(self.hops).__name__ not in ["int"] or self.hops < 0:
            raise ValueError(
                "CMN(collaborative Memory Networks)'s hops {} not supported, should be 'int'"
                "and greater than 0".format(
                    self.hops))

        if type(self.max_len).__name__ not in ["int"] or self.max_len < 0:
            raise ValueError(
                "CMN(collaborative Memory Networks)'s max_len {} not supported, should be 'int'"
                "and greater than 0".format(
                    self.max_len))

        if type(self.neg_count).__name__ not in ["int"] or self.max_len < 0:
            raise ValueError(
                "CMN(collaborative Memory Networks)'s neg_count {} not supported, should be 'int'"
                "and greater than 0".format(
                    self.neg_count))

        if type(self.l2_coef).__name__ not in ["int", "float"] and self.l2_coef < 0:
            raise ValueError(
                "CMN(collaborative Memory Networks)'s l2_coef {} not supported, should be 'int' or 'float'"
                "and greater than 0".format(
                    self.l2_coef))

        return True

    def _parse_early_stop(self, param):
        """
           Examples:
                "early_stop": {
                       "early_stop": "diff",
                       "eps": 0.0001
                   }
        """
        default_eps = 0.0001
        if isinstance(param, dict):
            early_stop = param.get("early_stop", None)
            eps = param.get("eps", default_eps)
            if not early_stop:
                raise ValueError(f"early_stop config: {param} invalid")
            return SimpleNamespace(converge_func=early_stop, eps=eps)
        else:
            raise ValueError(f"invalid type for early_stop: {type(param)}")

    def _parse_optimizer(self, param):
        """
        Examples:
            "optimize": {
                    "optimizer": "SGD",
                    "learning_rate": 0.05
            }
        """
        kwargs = {}
        if isinstance(param, str):
            return SimpleNamespace(optimizer=param, kwargs=kwargs)
        elif isinstance(param, dict):
            optimizer = param.get("optimizer", kwargs)
            if not optimizer:
                raise ValueError(f"optimizer config: {param} invalid")
            kwargs = {k: v for k, v in param.items() if k != "optimizer"}
            return SimpleNamespace(optimizer=optimizer, kwargs=kwargs)
        else:
            raise ValueError(f"invalid type for optimize: {type(param)}")

    def _parse_metrics(self, param):
        """
        Examples:

            1. "metrics": "Accuracy"
            2. "metrics": ["Accuracy"]
        """
        if not param:
            return []
        elif isinstance(param, str):
            return [param]
        elif isinstance(param, list):
            return param
        else:
            raise ValueError(f"invalid metrics type: {type(param)}")


class HeteroCMNParam(CMNParam):
    """
    Parameters
    ----------
    aggregate_iters : int, default: 1
        Indicate how many iterations are aggregated once.

    """

    def __init__(self, optimizer={"optimizer": "SGD", "learning_rate": 0.01},
                 early_stop={"early_stop": "diff"},
                 batch_size=-1, init_param=CMNInitParam(),
                 max_iter=100,
                 predict_param=PredictParam(), cv_param=CrossValidationParam(),
                 aggregate_iters=1, validation_freqs=None,
                 hops=2, max_len=4, l2_coef=0.1, neg_count=10,
                 secure_aggregate: bool = True,
                 aggregate_every_n_epoch: int = 1,
                 loss: str = "mse",
                 metrics: str = "mse"):
        super(HeteroCMNParam, self).__init__(optimizer=optimizer,
                                             batch_size=batch_size,
                                             init_param=init_param, max_iter=max_iter,
                                             early_stop=early_stop,
                                             predict_param=predict_param,
                                             cv_param=cv_param,
                                             validation_freqs=validation_freqs,
                                             hops=hops,
                                             max_len=max_len,
                                             neg_count=neg_count,
                                             l2_coef=l2_coef,
                                             secure_aggregate=secure_aggregate,
                                             aggregate_every_n_epoch=aggregate_every_n_epoch,
                                             loss=loss,
                                             metrics=metrics)
        self.aggregate_iters = aggregate_iters

    def check(self):
        super().check()

        if not isinstance(self.aggregate_iters, int):
            raise ValueError(
                "HeteroCMNParam's aggregate_iters {} not supported, should be int type".format(
                    self.aggregate_iters))

        return True

    def generate_pb(self):
        pb = cmn_model_meta_pb2.HeteroCMNParam()
        pb.secure_aggregate = self.secure_aggregate
        pb.aggregate_every_n_epoch = self.aggregate_every_n_epoch
        pb.batch_size = self.batch_size
        pb.max_iter = self.max_iter
        pb.early_stop.early_stop = self.early_stop.converge_func
        pb.early_stop.eps = self.early_stop.eps
        pb.hops = self.hops
        pb.max_len = self.max_len
        pb.l2_coef = self.l2_coef

        for metric in self.metrics:
            pb.metrics.append(metric)

        pb.optimizer.optimizer = self.optimizer.optimizer
        pb.optimizer.args = json.dumps(self.optimizer.kwargs)
        pb.loss = self.loss
        pb.embed_dim = self.init_param.embed_dim
        return pb

    def restore_from_pb(self, pb):
        self.secure_aggregate = pb.secure_aggregate
        self.aggregate_every_n_epoch = pb.aggregate_every_n_epoch
        self.batch_size = pb.batch_size
        self.max_iter = pb.max_iter
        self.early_stop = self._parse_early_stop(dict(early_stop=pb.early_stop.early_stop, eps=pb.early_stop.eps))
        self.metrics = list(pb.metrics)
        self.optimizer = self._parse_optimizer(dict(optimizer=pb.optimizer.optimizer, **json.loads(pb.optimizer.args)))
        self.loss = pb.loss
        self.max_len = pb.max_len
        self.hops = pb.hops
        self.l2_coef = pb.l2_coef
        return pb
