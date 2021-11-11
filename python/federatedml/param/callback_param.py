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
from federatedml.param.base_param import BaseParam


class CallbackParam(BaseParam):
    """
    Define callback method that used in federated ml.

    Parameters
    ----------
    callbacks : list, default: []
        Indicate what kinds of callback functions is desired during the training process.
        Accepted values: {'EarlyStopping', 'ModelCheckpoint'， 'PerformanceEvaluate'}

    validation_freqs: {None, int, list, tuple, set}
        validation frequency during training.

    early_stopping_rounds: None or int
        Will stop training if one metric doesn’t improve in last early_stopping_round rounds

    metrics: None, or list
        Indicate when executing evaluation during train process, which metrics will be used. If set as empty,
        default metrics for specific task type will be used. As for binary classification, default metrics are
        ['auc', 'ks']

    use_first_metric_only: bool, default: False
        Indicate whether use the first metric only for early stopping judgement.

    save_freq: int, default: 1
        The callbacks save model every save_freq epoch


    """

    def __init__(self, callbacks=None, validation_freqs=None, early_stopping_rounds=None,
                 metrics=None, use_first_metric_only=False, save_freq=1):
        super(CallbackParam, self).__init__()
        self.callbacks = callbacks or []
        self.validation_freqs = validation_freqs
        self.early_stopping_rounds = early_stopping_rounds
        self.metrics = metrics or []
        self.use_first_metric_only = use_first_metric_only
        self.save_freq = save_freq

    def check(self):

        if self.early_stopping_rounds is None:
            pass
        elif isinstance(self.early_stopping_rounds, int):
            if self.early_stopping_rounds < 1:
                raise ValueError("early stopping rounds should be larger than 0 when it's integer")
            if self.validation_freqs is None:
                raise ValueError("validation freqs must be set when early stopping is enabled")

        if self.validation_freqs is not None:
            if type(self.validation_freqs).__name__ not in ["int", "list", "tuple", "set"]:
                raise ValueError(
                    "validation strategy param's validate_freqs's type not supported ,"
                    " should be int or list or tuple or set"
                )
            if type(self.validation_freqs).__name__ == "int" and \
                    self.validation_freqs <= 0:
                raise ValueError("validation strategy param's validate_freqs should greater than 0")

        if self.metrics is not None and not isinstance(self.metrics, list):
            raise ValueError("metrics should be a list")

        if not isinstance(self.use_first_metric_only, bool):
            raise ValueError("use_first_metric_only should be a boolean")

        return True
