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
from arch.api import federation
from arch.api.utils import log_utils
from federatedml.logistic_regression.hetero_logistic_regression import HeteroLRGuest
from federatedml.model_selection import MiniBatch
from federatedml.optim import activation
from federatedml.optim.gradient import HeteroLogisticGradient
from federatedml.util import consts
from federatedml.util.transfer_variable import HeteroLRTransferVariable

LOGGER = log_utils.getLogger()


class HeteroDNNLRGuest(HeteroLRGuest):
    def __init__(self, local_model, logistic_params):
        super(HeteroDNNLRGuest, self).__init__(logistic_params)

        self.localModel = local_model
        self.feature_dim = local_model.get_encode_dim()

    def transform(self, batch_data_inst):
        return batch_data_inst

    def update_local_model(self, fore_gradient, coef):
        pass
