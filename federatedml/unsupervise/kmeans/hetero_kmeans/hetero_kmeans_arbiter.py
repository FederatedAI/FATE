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

    def sum_in_cluster(self,iterator):
        sum_result = dict()
        result = []
        for k,v in iterator:
            if v[1] not in sum_result:
                sum_result[v[1]] = np.sqrt(v[0][v[1]])
            else:
                sum_result[v[1]] += np.sqrt(v[0][v[1]])
        for i in range(len(sum_result)):
            result[i] = sum_result
        return result

    def cal_ave_dist(self, dist_cluster_dtable, cluster_result, k):
        dist_centroid_dist_dtable = dist_cluster_dtable.mapPartitions(self.sum_in_cluster).reduce(self.sum_dict)
        cluster_count = cluster_result.mapPartitions(self.count).reduce(self.sum_dict)
        cal_ave_dist_list = []
        for k in dist_centroid_dist_dtable:
            count = cluster_count[k]
            cal_ave_dist_list.append(dist_centroid_dist_dtable[k] / count)
        return cal_ave_dist_list


    # def centroid_assign(self, dist_sum):
    #     new_centroid= []
    #     for row in dist_sum:
    #         new_centroid.append(np.argmin(row))
    #     return new_centroid

    def fit(self, data_instances=None):
        LOGGER.info("Enter hetero Kmeans arbiter fit")
        while self.n_iter_ < self.max_iter:
            secure_dist_all_1 = self.transfer_variable.guest_dist.get(idx=0, suffix=(self.n_iter_,))
            secure_dist_all_2 = self.transfer_variable.host_dist.get(idx=0, suffix=(self.n_iter_,))
            dist_sum = secure_dist_all_1.join(secure_dist_all_2, lambda v1, v2: v1+v2)
            #dist_sum = self.dist_aggregator.sum_model(suffix=(self.n_iter_,))
            cluster_result = dist_sum.mapValues(lambda k, v: np.argmin(v))
            self.transfer_variable.cluster_result.remote(cluster_result, role=consts.GUEST, idx=0, suffix=(self.n_iter_,))
            self.transfer_variable.cluster_result.remote(cluster_result, role=consts.HOST, idx=-1, suffix=(self.n_iter_,))

            dist_cluster_dtable = dist_sum.join(cluster_result, lambda v1, v2: [v1, v2])
            dist_table = self.cal_ave_dist(dist_cluster_dtable, cluster_result, self.k)#ave dist in each cluster
            cluster_dist = self.cluster_dist_aggregator.sum_model(suffix=(self.n_iter_,))
            self.DBI=clustering_metric.Davies_Bouldin_index.compute(dist_table, cluster_dist)

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
        result=[]
        for i in range(len(dist_sum)):
            item = tuple(i,[-1,sample_class[i],dist_table,cluster_dist])
            result.append(item)
        predict_result = session.parallelize(result)
        return predict_result
