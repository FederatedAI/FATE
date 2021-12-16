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

from federatedml.callbacks.validation_strategy import ValidationStrategy
from federatedml.callbacks.model_checkpoint import ModelCheckpoint
from federatedml.param.callback_param import CallbackParam
from federatedml.util import LOGGER


class CallbackList(object):
    def __init__(self, role, mode, model):
        self.role = role
        self.mode = mode
        self.model = model
        self.callback_list = []

    def init_callback_list(self, callback_param: CallbackParam):
        LOGGER.debug(f"self_model: {self.model}")

        if "EarlyStopping" in callback_param.callbacks or \
                "PerformanceEvaluate" in callback_param.callbacks:
            has_arbiter = self.model.component_properties.has_arbiter
            validation_strategy = ValidationStrategy(self.role, self.mode,
                                                     callback_param.validation_freqs,
                                                     callback_param.early_stopping_rounds,
                                                     callback_param.use_first_metric_only,
                                                     arbiter_comm=has_arbiter)
            self.callback_list.append(validation_strategy)
        if "ModelCheckpoint" in callback_param.callbacks:
            model_checkpoint = ModelCheckpoint(model=self.model,
                                               save_freq=callback_param.save_freq)
            self.callback_list.append(model_checkpoint)

    def get_validation_strategy(self):
        for callback_func in self.callback_list:
            if isinstance(callback_func, ValidationStrategy):
                return callback_func
        return None

    def on_train_begin(self, train_data=None, validate_data=None):
        for callback_func in self.callback_list:
            callback_func.on_train_begin(train_data, validate_data)

    def on_epoch_end(self, epoch):
        for callback_func in self.callback_list:
            callback_func.on_epoch_end(self.model, epoch)

    def on_epoch_begin(self, epoch):
        for callback_func in self.callback_list:
            callback_func.on_epoch_begin(self.model, epoch)

    def on_train_end(self):
        for callback_func in self.callback_list:
            callback_func.on_train_end(self.model)
