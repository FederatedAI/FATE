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

from numpy import *
from arch.api.utils import log_utils
from federatedml.unsupervise.kmeans.kmeans_model_base import BaseKmeansModel
from federatedml.param.hetero_kmeans_param import KmeansParam


LOGGER = log_utils.getLogger()


class HeteroKmenasArbiter(BaseKmeansModel):
    def __init__(self):
        super(HeteroKmenasArbiter, self).__init__()
        self.model_param = KmeansParam()

    def fit(self, data_instances=None):
        LOGGER.info("Enter hetero linear model arbiter fit")
        n= inf
        while self.n_iter_ < self.max_iter and n > self.tol:
            for i in range(0, self.k):
                k1 = self.transfer_variable.guest_dist.get(idx= -1, suffix=self.n_iter_)
            self.n_iter_ += 1
