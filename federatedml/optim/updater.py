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
# from arch.api.utils import log_utils

# LOGGER = log_utils.getLogger()


class Updater:
    def __init__(self, alpha, learning_rate=0.01):
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.shrinkage_val = self.alpha * self.learning_rate

    def loss_norm(self, coef_): pass

    def update_coef(self, coef_, gradient): pass


class L1Updater(Updater):
    def loss_norm(self, coef_):
        return np.sum(self.alpha * np.abs(coef_))

    def update_coef(self, coef_, gradient):
        return np.sign(coef_ - gradient) * np.maximum(0, np.abs(coef_ - gradient) - self.shrinkage_val)


class L2Updater(Updater):
    def loss_norm(self, coef_):
        return 0.5 * self.alpha * np.dot(coef_, coef_)

    def update_coef(self, coef_, gradient):
        return coef_ - gradient - self.learning_rate * self.alpha * coef_
