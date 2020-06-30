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
from federatedml.linear_model.logistic_regression.hetero_logistic_regression.hetero_lr_base import HeteroLRBase
from federatedml.optim.gradient import hetero_lr_gradient_and_loss
from federatedml.param.logistic_regression_param import HeteroLogisticParam
from federatedml.util import consts


LOGGER = log_utils.getLogger()


class HeteroLRArbiter(HeteroBaseArbiter, HeteroLRBase):
    def __init__(self):
        super(HeteroLRArbiter, self).__init__()
        self.gradient_loss_operator = hetero_lr_gradient_and_loss.Arbiter()
        self.model_param = HeteroLogisticParam()
        self.n_iter_ = 0
        self.header = None
        self.is_converged = False
        self.model_param_name = 'HeteroLogisticRegressionParam'
        self.model_meta_name = 'HeteroLogisticRegressionMeta'
        self.model_name = 'HeteroLogisticRegression'
        self.need_one_vs_rest = None
        self.need_call_back_loss = True
        self.mode = consts.HETERO

    def fit(self, data_instances=None, validate_data=None):
        LOGGER.debug("Has loss_history: {}".format(hasattr(self, 'loss_history')))
        LOGGER.debug("Need one_vs_rest: {}".format(self.need_one_vs_rest))
        classes = self.one_vs_rest_obj.get_data_classes(data_instances)
        if len(classes) > 2:
            self.need_one_vs_rest = True
            self.need_call_back_loss = False
            self.one_vs_rest_fit(train_data=data_instances, validate_data=validate_data)
        else:
            self.need_one_vs_rest = False
            super().fit(data_instances, validate_data)

    def fit_binary(self, data_instances, validate_data):
        super().fit(data_instances, validate_data)
