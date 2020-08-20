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
from arch.api import session
from federatedml.evaluation.metrics import clustering_metric


LOGGER = log_utils.getLogger()


class HeteroKmeansArbiter(BaseKmeansModel):
    def __init__(self):
        super(HeteroKmeansArbiter, self).__init__()
        self.model_param = KmeansParam()
        self.dist_aggregator = secure_sum_aggregator.Server(enable_secure_aggregate=True)
        self.cluster_dist_aggregator = secure_sum_aggregator.Server(enable_secure_aggregate=True)
        self.DBI = 0

    def cal_ave_dist(self, dist_sum, cluster_result, k):
        for i in range(0, k):
            dist_list =[]
            for j in range(len(dist_sum)):
                sum = 0
                count = 0
                if cluster_result[j]== i:
                    sum += dist_sum[j][i]
                    count += 1
                ave_dist = sum / count
            dist_list.append(ave_dist)
        return dist_list

    def fit(self, data_instances=None):
        LOGGER.info("Enter hetero Kmeans arbiter fit")
        while self.n_iter_ < self.max_iter:
            dist_sum = self.dist_aggregator.sum_model(suffix=(self.n_iter_,))
            cluster_result = self.centroid_assign(dist_sum)
            self.transfer_variable.cluster_result.remote(cluster_result, role=consts.GUEST, idx=0, suffix=(self.n_iter_,))
            self.transfer_variable.cluster_result.remote(cluster_result, role=consts.HOST, idx=-1, suffix=(self.n_iter_,))

            dist_table = self.cal_ave_dist(dist_sum, cluster_result, self.k)
            cluster_dist = self.cluster_dist_aggregator.sum_model(suffix=(self.n_iter_,))
            self.DBI=clustering_metric.AdjustedRandScore.compute(dist_table, cluster_dist)

            tol1 = self.transfer_variable.guest_tol.get(idx=0, suffix=(self.n_iter_,))
            tol2 = self.transfer_variable.host_tol.get(idx=-1, suffix=(self.n_iter_,))
            tol_final = tol1+tol2
            tol_tag = 0 if tol_final > self.tol else 1
            self.transfer_variable.arbiter_tol.remote(tol_tag, role=consts.HOST, idx=-1, suffix=(self.n_iter_,))
            self.transfer_variable.arbiter_tol.remote(tol_tag, role=consts.GUEST, idx=0, suffix=(self.n_iter_,))
            if tol_tag:
                break

            self.n_iter_ += 1

    def predict(self, data_instances):
        LOGGER.info("Start predict ...")
        dist_sum = self.dist_aggregator.sum_model(suffix='predict')
        sample_class = self.centroid_assign(dist_sum)
        dist_table = self.cal_ave_dist(dist_sum, sample_class, self.k)
        cluster_dist = self.cluster_dist_aggregator.sum_model(suffix='predict')
        self.transfer_variable.cluster_result.remote(sample_class, role=consts.Guest, idx=0)

        return