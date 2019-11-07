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

from arch.api.utils import log_utils
from federatedml.model_selection.k_fold import KFold

LOGGER = log_utils.getLogger()


def _get_cv_param(model):
    model.model_param.cv_param.role = model.role
    model.model_param.cv_param.mode = model.mode
    return model.model_param.cv_param


def run(model, data_instances):
    if not model.need_run:
        return data_instances
    kflod_obj = KFold()
    cv_param = _get_cv_param(model)
    kflod_obj.run(cv_param, data_instances, model)
    LOGGER.info("Finish KFold run")
    return data_instances


class SomeModel(ModelBase):

    def __init__(self):
        self.sklearn_obj = None # 自定义一下

    def fit(self, data_instances):
        X,y = self.load(data_instances)
        self.sklearn_obj.fit(X, y)

    def predict(self, data_instances):
        X,y = self.load(data_instances)
        predict_result = self.sklearn_obj.predict(X, y)
        return self.transfer_predict_result(predict_result)  # 转成现在用的格式

    def run_cv(self, data_instances):
        kflod_obj = KFold()
        kflod_obj.mode = 'unilateral'  #  避免alignment
        cv_param = _get_cv_param(self)
        kflod_obj.run(cv_param, data_instances, self)
        LOGGER.info("Finish KFold run")
        return data_instances
