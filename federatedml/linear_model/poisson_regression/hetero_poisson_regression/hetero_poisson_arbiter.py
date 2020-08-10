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
from federatedml.linear_model.poisson_regression.hetero_poisson_regression.hetero_poisson_base import HeteroPoissonBase
from federatedml.optim.gradient import hetero_poisson_gradient_and_loss
from federatedml.param.poisson_regression_param import PoissonParam
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class HeteroPoissonArbiter(HeteroBaseArbiter, HeteroPoissonBase):
    def __init__(self):
        super(HeteroPoissonArbiter, self).__init__()
        self.gradient_loss_operator = hetero_poisson_gradient_and_loss.Arbiter()
        self.model_param = PoissonParam()
        self.n_iter_ = 0
        self.header = None
        self.model_param_name = 'HeteroPoissonRegressionParam'
        self.model_meta_name = 'HeteroPoissonRegressionMeta'
        self.model_name = 'HeteroPoissonRegression'
        self.is_converged = False
        self.mode = consts.HETERO
        self.need_call_back_loss = True
