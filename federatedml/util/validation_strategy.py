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
        if not self.validation_freqs:
            return False

        if isinstance(self.validation_freqs, int):
            return (epoch + 1) % self.validation_freqs == 0

        return epoch + 1 in self.validation_freqs

    def generate_flowid(self, prefix, epoch, keywords="iteration"):
        return "_".join([prefix, keywords, str(epoch)])

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

        model_flowid = model.flowid
        flowid = self.generate_flowid(model_flowid, epoch)
        model.flowid = flowid
        predicts = model.predict(data)
        model.flowid = model_flowid

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
            predicts = train_predicts
            if validate_predicts:
                predicts = predicts.union(validate_predicts)

            self.evaluate(predicts, model, epoch)

