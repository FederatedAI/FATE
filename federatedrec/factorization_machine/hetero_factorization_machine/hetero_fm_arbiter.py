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
from federatedml.util import consts
from federatedrec.optim.gradient import hetero_fm_gradient_and_loss
from federatedrec.param.factorization_machine_param import HeteroFactorizationParam
from federatedrec.factorization_machine.base_fm_model_arbiter import HeteroBaseArbiter
from federatedrec.factorization_machine.hetero_factorization_machine.hetero_fm_base import HeteroFMBase


LOGGER = log_utils.getLogger()


class HeteroFMArbiter(HeteroBaseArbiter, HeteroFMBase):
    def __init__(self):
        super(HeteroFMArbiter, self).__init__()
        self.gradient_loss_operator = hetero_fm_gradient_and_loss.Arbiter()
        self.model_param = HeteroFactorizationParam()
        self.n_iter_ = 0
        self.header = None
        self.is_converged = False
        self.model_param_name = 'HeteroFactorizationMachineParam'
        self.model_meta_name = 'HeteroFactorizationMachineMeta'
        self.need_one_vs_rest = None
        self.in_one_vs_rest = False
        self.mode = consts.HETERO

    def fit(self, data_instances=None, validate_data=None):
        LOGGER.debug("Need one_vs_rest: {}".format(self.need_one_vs_rest))
        classes = self.one_vs_rest_obj.get_data_classes(data_instances)
        if len(classes) > 2:
            self.need_one_vs_rest = True
            self.in_one_vs_rest = True
            self.one_vs_rest_fit(train_data=data_instances, validate_data=validate_data)
        else:
            self.need_one_vs_rest = False
            super().fit(data_instances, validate_data)

    def fit_binary(self, data_instances, validate_data):
        super().fit(data_instances, validate_data)
