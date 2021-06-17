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

from abc import ABC
import abc
from federatedml.ensemble.boosting.boosting_core import Boosting
from federatedml.param.boosting_param import HeteroBoostingParam
from federatedml.secureprotol import IterativeAffineEncrypt
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.encrypt_mode import EncryptModeCalculator
from federatedml.util import consts
from federatedml.feature.binning.quantile_binning import QuantileBinning
from federatedml.util.classify_label_checker import ClassifyLabelChecker
from federatedml.util.classify_label_checker import RegressionLabelChecker
from federatedml.util import LOGGER
from fate_flow.entity.metric import Metric
from fate_flow.entity.metric import MetricMeta
from federatedml.transfer_variable.transfer_class.hetero_boosting_transfer_variable import \
    HeteroBoostingTransferVariable
from federatedml.util.io_check import assert_io_num_rows_equal


class HeteroBoosting(Boosting, ABC):

    def __init__(self):
        super(HeteroBoosting, self).__init__()
        self.encrypter = None
        self.encrypted_calculator = None
        self.early_stopping_rounds = None
        self.binning_class = QuantileBinning
        self.model_param = HeteroBoostingParam()
        self.transfer_variable = HeteroBoostingTransferVariable()
        self.mode = consts.HETERO

    def _init_model(self, param: HeteroBoostingParam):
        LOGGER.debug('in hetero boosting, objective param is {}'.format(param.objective_param.objective))
        super(HeteroBoosting, self)._init_model(param)
        self.encrypt_param = param.encrypt_param
        self.re_encrypt_rate = param.encrypted_mode_calculator_param
        self.calculated_mode = param.encrypted_mode_calculator_param.mode
        self.re_encrypted_rate = param.encrypted_mode_calculator_param.re_encrypted_rate
        self.early_stopping_rounds = param.early_stopping_rounds
        self.use_first_metric_only = param.use_first_metric_only

    def generate_encrypter(self):
        LOGGER.info("generate encrypter")
        if self.encrypt_param.method.lower() == consts.PAILLIER.lower():
            self.encrypter = PaillierEncrypt()
            self.encrypter.generate_key(self.encrypt_param.key_length)
        elif self.encrypt_param.method.lower() == consts.ITERATIVEAFFINE.lower():
            self.encrypter = IterativeAffineEncrypt()
            self.encrypter.generate_key(key_size=self.encrypt_param.key_length,
                                        randomized=False)
        elif self.encrypt_param.method.lower() == consts.RANDOM_ITERATIVEAFFINE.lower():
            self.encrypter = IterativeAffineEncrypt()
            self.encrypter.generate_key(key_size=self.encrypt_param.key_length,
                                        randomized=True)
        else:
            raise NotImplementedError("encrypt method not supported yes!!!")

        self.encrypted_calculator = EncryptModeCalculator(self.encrypter, self.calculated_mode, self.re_encrypted_rate)

    def check_label(self):

        LOGGER.info("check label")
        classes_ = []
        num_classes, booster_dim = 1, 1
        if self.task_type == consts.CLASSIFICATION:
            num_classes, classes_ = ClassifyLabelChecker.validate_label(self.data_bin)
            if num_classes > 2:
                booster_dim = num_classes

            range_from_zero = True
            for _class in classes_:
                try:
                    if 0 <= _class < len(classes_) and isinstance(_class, int):
                        continue
                    else:
                        range_from_zero = False
                        break
                except:
                    range_from_zero = False

            classes_ = sorted(classes_)
            if not range_from_zero:
                class_mapping = dict(zip(classes_, range(num_classes)))
                self.y = self.y.mapValues(lambda _class: class_mapping[_class])

        else:
            RegressionLabelChecker.validate_label(self.data_bin)

        return classes_, num_classes, booster_dim


class HeteroBoostingGuest(HeteroBoosting, ABC):

    def __init__(self):
        super(HeteroBoostingGuest, self).__init__()

    def _init_model(self, param):
        super(HeteroBoostingGuest, self)._init_model(param)

    def sync_booster_dim(self):
        LOGGER.info("sync booster_dim to host")

        self.transfer_variable.booster_dim.remote(self.booster_dim,
                                                  role=consts.HOST,
                                                  idx=-1)

    def sync_stop_flag(self, stop_flag, num_round):
        LOGGER.info("sync stop flag to host, boosting_core round is {}".format(num_round))

        self.transfer_variable.stop_flag.remote(stop_flag,
                                                role=consts.HOST,
                                                idx=-1,
                                                suffix=(num_round,))

    def sync_predict_round(self, predict_round,):
        LOGGER.info("sync predict start round {}".format(predict_round))
        self.transfer_variable.predict_start_round.remote(predict_round, role=consts.HOST, idx=-1,)

    def fit(self, data_inst, validate_data=None):

        LOGGER.info('begin to fit a hetero boosting model, model is {}'.format(self.model_name))

        self.data_inst = data_inst

        self.data_bin, self.bin_split_points, self.bin_sparse_points = self.prepare_data(data_inst)

        self.y = self.get_label(self.data_bin)

        self.classes_, self.num_classes, self.booster_dim = self.check_label()

        LOGGER.info('class index is {}'.format(self.classes_))

        self.loss = self.get_loss_function()

        self.sync_booster_dim()

        self.y_hat, self.init_score = self.get_init_score(self.y, self.num_classes)

        self.generate_encrypter()

        self.callback_meta("loss",
                           "train",
                           MetricMeta(name="train",
                                      metric_type="LOSS",
                                      extra_metas={"unit_name": "iters"}))

        self.validation_strategy = self.init_validation_strategy(data_inst, validate_data)

        for epoch_idx in range(self.boosting_round):

            LOGGER.info('cur epoch idx is {}'.format(epoch_idx))

            for class_idx in range(self.booster_dim):

                # fit a booster
                model = self.fit_a_booster(epoch_idx, class_idx)

                booster_meta, booster_param = model.get_model()

                if booster_meta is not None and booster_param is not None:
                    self.booster_meta = booster_meta
                    self.boosting_model_list.append(booster_param)

                # update predict score
                cur_sample_weights = model.get_sample_weights()
                self.y_hat = self.get_new_predict_score(self.y_hat, cur_sample_weights, dim=class_idx)

            # compute loss
            loss = self.compute_loss(self.y_hat, self.y)
            self.history_loss.append(loss)
            LOGGER.info("round {} loss is {}".format(epoch_idx, loss))
            self.callback_metric("loss",
                                 "train",
                                 [Metric(epoch_idx, loss)])

            if self.validation_strategy:
                self.validation_strategy.validate(self, epoch_idx, use_precomputed_train=True,
                                                  train_scores=self.score_to_predict_result(data_inst, self.y_hat))

            should_stop_a, should_stop_b = False, False
            if self.validation_strategy is not None:
                if self.validation_strategy.need_stop():
                    should_stop_a = True

            if self.n_iter_no_change and self.check_convergence(loss):
                should_stop_b = True
                self.is_converged = True

            self.sync_stop_flag(self.is_converged, epoch_idx)

            if should_stop_a or should_stop_b:
                break

        self.callback_meta("loss",
                           "train",
                           MetricMeta(name="train",
                                      metric_type="LOSS",
                                      extra_metas={"Best": min(self.history_loss)}))

        if self.validation_strategy and self.validation_strategy.has_saved_best_model():
            LOGGER.info('best model exported')
            self.load_model(self.validation_strategy.cur_best_model)

        # get summary
        self.set_summary(self.generate_summary())

    @assert_io_num_rows_equal
    def predict(self, data_inst):
        # predict is implemented in hetero_secureboost
        raise NotImplementedError('predict func is not implemented')

    @abc.abstractmethod
    def fit_a_booster(self, epoch_idx: int, booster_dim: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_booster(self, model_meta, model_param, epoch_idx, booster_idx):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_model_meta(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_model_param(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_model_meta(self, model_meta):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_model_param(self, model_param):
        raise NotImplementedError()


class HeteroBoostingHost(HeteroBoosting, ABC):

    def __init__(self):
        super(HeteroBoostingHost, self).__init__()

    def _init_model(self, param):
        super(HeteroBoostingHost, self)._init_model(param)

    def sync_booster_dim(self):
        LOGGER.info("sync booster dim from guest")
        self.booster_dim = self.transfer_variable.booster_dim.get(idx=0)
        LOGGER.info("booster dim is %d" % self.booster_dim)

    def sync_stop_flag(self, num_round):
        LOGGER.info("sync stop flag from guest, boosting_core round is {}".format(num_round))
        stop_flag = self.transfer_variable.stop_flag.get(idx=0,
                                                         suffix=(num_round,))
        return stop_flag

    def sync_predict_start_round(self,):
        return self.transfer_variable.predict_start_round.get(idx=0,)

    def fit(self, data_inst, validate_data=None):

        LOGGER.info('begin to fit a hetero boosting model, model is {}'.format(self.model_name))

        self.data_bin, self.bin_split_points, self.bin_sparse_points = self.prepare_data(data_inst)
        self.sync_booster_dim()
        self.generate_encrypter()

        self.validation_strategy = self.init_validation_strategy(data_inst, validate_data)

        for epoch_idx in range(self.boosting_round):

            LOGGER.info('cur epoch idx is {}'.format(epoch_idx))

            for class_idx in range(self.booster_dim):
                # fit a booster
                model = self.fit_a_booster(epoch_idx, class_idx)  # need to implement
                booster_meta, booster_param = model.get_model()
                if booster_meta is not None and booster_param is not None:
                    self.booster_meta = booster_meta
                    self.boosting_model_list.append(booster_param)

            if self.validation_strategy:
                self.validation_strategy.validate(self, epoch_idx, use_precomputed_train=True, train_scores=None)

            should_stop_a = False
            if self.validation_strategy is not None:
                if self.validation_strategy.need_stop():
                    should_stop_a = True

            should_stop_b = self.sync_stop_flag(epoch_idx)
            self.is_converged = should_stop_b
            if should_stop_a or should_stop_b:
                break

        if self.validation_strategy and self.validation_strategy.has_saved_best_model():
            LOGGER.info('best model exported')
            self.load_model(self.validation_strategy.cur_best_model)

        self.set_summary(self.generate_summary())

    def lazy_predict(self, data_inst):

        LOGGER.info('running guest lazy prediction')
        data_inst = self.data_alignment(data_inst)
        init_score = self.init_score
        self.predict_y_hat = data_inst.mapValues(lambda v: init_score)

        rounds = len(self.boosting_model_list) // self.booster_dim
        predict_start_round = self.sync_predict_start_round()

        for idx in range(predict_start_round, rounds):
            for booster_idx in range(self.booster_dim):
                model = self.load_booster(self.booster_meta,
                                          self.boosting_model_list[idx * self.booster_dim + booster_idx],
                                          idx, booster_idx)
                model.predict(data_inst)

        LOGGER.debug('lazy prediction finished')

    def predict(self, data_inst):

        LOGGER.info('using default lazy prediction')
        self.lazy_predict(data_inst)

    @abc.abstractmethod
    def load_booster(self, model_meta, model_param, epoch_idx, booster_idx):
        raise NotImplementedError()

    @abc.abstractmethod
    def fit_a_booster(self, epoch_idx: int, booster_dim: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_model_meta(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_model_param(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_model_meta(self, model_meta):
        raise NotImplementedError()

    @abc.abstractmethod
    def set_model_param(self, model_param):
        raise NotImplementedError()
