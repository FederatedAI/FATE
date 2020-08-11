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
from federatedml.util import consts
from federatedml.framework.homo.blocks import secure_sum_aggregator

LOGGER = log_utils.getLogger()


class HeteroKmenasArbiter(BaseKmeansModel):
    def __init__(self):
        super(HeteroKmenasArbiter, self).__init__()
        self.model_param = KmeansParam()
        self.dist_aggregator = secure_sum_aggregator.Server(enable_secure_aggregate=True)

    def fit(self, data_instances=None):
        LOGGER.info("Enter hetero Kmeans arbiter fit")
        tol_sum = inf
        while self.n_iter_ < self.max_iter :
            # p1 = self.transfer_variable.guest_dist.get(idx=0, suffix=(self.n_iter_,))
            # p2 = self.transfer_variable.host_dist.get(idx=-1, suffix=(self.n_iter_,))
            dist_sum = self.dist_aggregator.sum_model(suffix=(self.n_iter_,))
            # dist_sum = p1
            # for p in p2:
            #     dist_sum = dist_sum.join(p, lambda v1, v2: np.array(v1) + np.array(v2))

            new_centroid = self.centroid_assign(dist_sum)
            self.transfer_variable.cluster_result.remote(new_centroid, role=consts.GUEST, idx=0, suffix=(self.n_iter_,))
            self.transfer_variable.cluster_result.remote(new_centroid, role=consts.HOST, idx=-1, suffix=(self.n_iter_,))

            tol1 = self.transfer_variable.guest_tol.get(idx=0, suffix=(self.n_iter_,))
            tol2 = self.transfer_variable.host_tol.get(idx=-1, suffix=(self.n_iter_,))
            tol_sum = tol1
            for tol in tol2:
                tol_sum += tol2[tol]
            tol_final = np.sum(tol_sum**2)**0.5

            if tol_final < self.tol:
                tol_tag = 1
                self.transfer_variable.arbiter_tol.remote(tol_tag, role=consts.HOST, idx=-1)
                self.transfer_variable.arbiter_tol.remote(tol_tag, role=consts.GUEST, idx=0)
                break

            self.n_iter_ += 1

    def predict(self, data_instances):
        LOGGER.info("Start predict ...")
        dist_sum = self.dist_aggregator.sum_model(suffix=1)
        sample_class = self.centroid_assign(dist_sum)
        self.transfer_variable.cluster_result.remote(sample_class, role=consts.Guest, idx=0)