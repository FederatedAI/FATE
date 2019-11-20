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
################################################################################
#
#
################################################################################

from arch.api.utils import log_utils
from federatedml.util import consts
from federatedml.evaluation.evaluation import Evaluation

LOGGER = log_utils.getLogger()


class ValidationStrategy(object):
    """
    This module is used for evaluating the performance of model during training process.
        it will be called only in fit process of models.

    Attributes
    ----------

    validation_freqs: None or positive integer or container object in python. Do validation in training process or Not.
                      if equals None, will not do validation in train process;
                      if equals positive integer, will validate data every validation_freqs epochs passes;
                      if container object in python, will validate data if epochs belong to this container.
                        e.g. validation_freqs = [10, 15], will validate data when epoch equals to 10 and 15.
                      Default: None

    train_data: None or DTable,
                if train_data not equal to None, and judge need to validate data according to validation_freqs,
                training data will be used for evaluating

    validate_data: None or DTable,
                if validate_data not equal to None, and judge need to validate data according to validation_freqs,
                validate data will be used for evaluating
    """
    def __init__(self, role=None, mode=None, validation_freqs=None):
        self.validation_freqs = validation_freqs
        self.role = role
        self.mode = mode
        self.flowid = ''
        self.train_data = None
        self.validate_data = None

        LOGGER.debug("end to init validation_strategy, freqs is {}".format(self.validation_freqs))

    def set_train_data(self, train_data):
        self.train_data = train_data

    def set_validate_data(self, validate_data):
        self.validate_data = validate_data

    def set_flowid(self, flowid):
        self.flowid = flowid

    def need_run_validation(self, epoch):
        LOGGER.debug("validation_freqs is {}".format(self.validation_freqs))
        if not self.validation_freqs:
            return False

        if isinstance(self.validation_freqs, int):
            return (epoch + 1) % self.validation_freqs == 0

        return epoch in self.validation_freqs

    def generate_flowid(self, prefix, epoch, keywords="iteration", data_type="train"):
        return "_".join([prefix, keywords, str(epoch), data_type])

    def make_data_set_name(self, need_cv, model_flowid, epoch):
        data_iteration_name = "_".join(["iteration", str(epoch)])
        if not need_cv:
            return data_iteration_name

        cv_fold = "_".join(["fold", model_flowid.split(".", -1)[-1]])
        return ".".join([cv_fold, data_iteration_name])

    def evaluate(self, predicts, model, epoch):
        evaluate_param = model.get_metrics_param()
        eval_obj = Evaluation()
        LOGGER.debug("evaluate type is {}".format(evaluate_param.eval_type))
        eval_obj._init_model(evaluate_param)
        eval_obj.set_tracker(model.tracker)
        data_set_name = self.make_data_set_name(model.need_cv, model.flowid,  epoch);
        eval_data = {data_set_name : predicts}
        eval_obj.fit(eval_data)
        eval_obj.save_data()
        LOGGER.debug("end to eval")

    def evaluate_data(self, model, epoch, data, data_type):
        if not data:
            return

        LOGGER.debug("start to evaluate data {}".format(data_type))
        model_flowid = model.flowid
        # model_flowid = ".".join(model.flowid.split(".", -1)[1:])
        flowid = self.generate_flowid(model_flowid, epoch, "iteration", data_type)
        model.set_flowid(flowid)
        predicts = model.predict(data)
        model.set_flowid(model_flowid)

        if self.mode == consts.HOMO and self.role == consts.ARBITER:
            pass
        elif self.mode == consts.HETERO and self.role == consts.HOST:
            pass
        else:
            predicts = predicts.mapValues(lambda value: value + [data_type])

        return predicts

    def validate(self, model, epoch):
        LOGGER.debug("begin to check validate status, need_run_validation is {}".format(self.need_run_validation(epoch)))
        if not self.need_run_validation(epoch):
            return

        if self.mode == consts.HOMO and self.role == consts.ARBITER:
            return

        train_predicts = self.evaluate_data(model, epoch, self.train_data, "train")
        validate_predicts = self.evaluate_data(model, epoch, self.validate_data, "validate")
        if train_predicts is None and validate_predicts is None:
            return
        else:
            LOGGER.debug("train_predicts data is {}".format(list(train_predicts.collect())))
            predicts = train_predicts
            if validate_predicts:
                LOGGER.debug("validate_predicts data is {}".format(list(validate_predicts.collect())))
                predicts = predicts.union(validate_predicts)

            LOGGER.debug("predicts data is {}".format(list(predicts.collect())))
            self.evaluate(predicts, model, epoch)

