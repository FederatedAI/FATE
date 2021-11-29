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

from federatedml.callbacks.callback_base import CallbackBase
from federatedml.util import LOGGER


class ModelCheckpoint(CallbackBase):
    def __init__(self, model, save_freq):
        self.model = model
        if save_freq == "epoch":
            save_freq = 1
        self.save_freq = save_freq
        self.save_count = 0

    def add_checkpoint(self, step_index, step_name=None, to_save_model=None):
        step_name = step_name if step_name is not None else self.model.step_name

        to_save_model = to_save_model if to_save_model is not None else self.model.export_serialized_models()
        _checkpoint = self.model.checkpoint_manager.new_checkpoint(step_index=step_index, step_name=step_name)
        _checkpoint.save(to_save_model)
        LOGGER.debug(f"current checkpoint num: {self.model.checkpoint_manager.checkpoints_number}")
        return _checkpoint

    def on_epoch_end(self, model, epoch):
        if epoch % self.save_freq == 0:
            self.add_checkpoint(step_index=epoch)
            self.save_count += 1
