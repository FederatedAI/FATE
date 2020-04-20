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

from arch.api.utils import log_utils
from federatedml.linear_model.base_linear_model_arbiter import HeteroBaseArbiter
from federatedml.linear_model.linear_regression.hetero_linear_regression.hetero_linr_base import HeteroLinRBase
from federatedml.optim.gradient import hetero_linr_gradient_and_loss
from federatedml.param.linear_regression_param import LinearParam
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroLinRArbiter(HeteroBaseArbiter, HeteroLinRBase):
    def __init__(self):
        super(HeteroLinRArbiter, self).__init__()
        self.gradient_loss_operator = hetero_linr_gradient_and_loss.Arbiter()
        self.model_param = LinearParam()
        self.n_iter_ = 0
        self.header = None
        self.model_param_name = 'HeteroLinearRegressionParam'
        self.model_meta_name = 'HeteroLinearRegressionMeta'
        self.model_name = 'HeteroLinearRegression'
        self.is_converged = False
        self.mode = consts.HETERO
        self.need_call_back_loss = True
