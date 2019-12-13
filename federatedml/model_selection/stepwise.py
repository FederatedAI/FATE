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

import numpy as np
from federatedml.util import consts
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class Stepwise(object):

    def __init__(self):
        self.mode = None
        self.role = None
        self.forward = False
        self.backward = False
        self.best_list = []
        self.n_step = 0

    def _init_model(self, param):
        self.model_param = param
        self.mode = param.mode
        self.role = param.role
        self.score = param.score
        self.direction = param.direction
        self.p_enter = param.p_enter
        self.p_remove = param.p_remove
        self.max_step = param.max_step
        self._get_direction()

    def _get_direction(self):
        if self.direction == "forward":
            self.forward = True
        elif self.direction == "backward":
            self.backward = True
        elif self.direction == "both":
            self.forward = True
            self.backward = True
        else:
            LOGGER.warning("Wrong stepwise direction given.")
            return

    def run(self, component_parameters, train_data, validate_data, model):
        self._init_model(component_parameters)






