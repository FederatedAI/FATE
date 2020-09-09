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

from fate_arch.session import computing_session as session
from fate_flow.entity.metric import Metric
from fate_flow.entity.metric import MetricMeta
from federatedml.evaluation.metrics import clustering_metric
from federatedml.framework.homo.blocks import secure_sum_aggregator
from federatedml.framework.homo.procedure import table_aggregator
from federatedml.param.hetero_kmeans_param import KmeansParam
from federatedml.unsupervise.kmeans.kmeans_model_base import BaseKmeansModel
from federatedml.util import LOGGER
from federatedml.util import consts


class HeteroKmeansArbiter(BaseKmeansModel):
    def __init__(self):
        super(HeteroKmeansArbiter, self).__init__()
        self.model_param = KmeansParam()
        self.dist_aggregator = secure_sum_aggregator.Server(enable_secure_aggregate=True)
        self.cluster_dist_aggregator = secure_sum_aggregator.Server(enable_secure_aggregate=True)
        self.DBI = 0
        self.aggregator = table_aggregator.Arbiter()

    def callback_dbi(self, iter_num, dbi):
        metric_meta = MetricMeta(name='train',
                                 metric_type="DBI",
                                 extra_metas={
                                     "unit_name": "iters",
                                 })

        self.callback_meta(metric_name='DBI', metric_namespace='train', metric_meta=metric_meta)
        self.callback_metric(metric_name='DBI',
                             metric_namespace='train',
                             metric_data=[Metric(iter_num, dbi)])

    def sum_in_cluster(self, iterator):
        sum_result = dict()
        for k, v in iterator:
            if v[1] not in sum_result:
                sum_result[v[1]] = np.sqrt(v[0][v[1]])
            else:
                sum_result[v[1]] += np.sqrt(v[0][v[1]])
        return sum_result

    def cal_ave_dist(self, dist_cluster_dtable, cluster_result, k):
        dist_centroid_dist_dtable = dist_cluster_dtable.mapPartitions(self.sum_in_cluster).reduce(self.sum_dict)
        cluster_count = cluster_result.mapPartitions(self.count).reduce(self.sum_dict)
        cal_ave_dist_list = []
        for i in range(self.k):
            count = cluster_count[i]
            cal_ave_dist_list.append([i, count, dist_centroid_dist_dtable[i] / count])
        return cal_ave_dist_list

    @staticmethod
    def max_radius(iterator):
        radius_result = dict()
        for k, v in iterator:
            if v[0] not in radius_result:
                radius_result[v[0]] = v[1]
            elif v[1] >= radius_result[v[0]]:
                radius_result[v[0]] = v[1]
        return radius_result

    @staticmethod
    def get_max_radius(v1, v2):
        rs = {}
        for k1 in v1:
            rs[k1] = max(v1[k1], v2[k1])
        return rs

    def fit(self, data_instances=None):
        LOGGER.info("Enter hetero Kmeans arbiter fit")
        while self.n_iter_ < self.max_iter:
            # secure_dist_all_1 = self.transfer_variable.guest_dist.get(idx=0, suffix=(self.n_iter_,))
            # secure_dist_all_2 = self.transfer_variable.host_dist.get(idx=0, suffix=(self.n_iter_,))
            # dist_sum = secure_dist_all_1.join(secure_dist_all_2, lambda v1, v2: v1 + v2)

            dist_sum = self.aggregator.aggregate_tables(suffix=(self.n_iter_,))
            cluster_result = dist_sum.mapValues(lambda v: np.argmin(v))
            self.aggregator.send_aggregated_tables(cluster_result, suffix=(self.n_iter_,))

            # self.transfer_variable.cluster_result.remote(cluster_result, role=consts.GUEST, idx=0,
            #                                              suffix=(self.n_iter_,))
            # self.transfer_variable.cluster_result.remote(cluster_result, role=consts.HOST, idx=0,
            #                                              suffix=(self.n_iter_,))

            dist_cluster_dtable = dist_sum.join(cluster_result, lambda v1, v2: [v1, v2])
            dist_table = self.cal_ave_dist(dist_cluster_dtable, cluster_result, self.k)  # ave dist in each cluster
            cluster_dist = self.cluster_dist_aggregator.sum_model(suffix=(self.n_iter_,))
            cluster_avg_intra_dist = []
            for i in range(len(dist_table)):
                cluster_avg_intra_dist.append(dist_table[i][2])
            self.DBI = clustering_metric.DaviesBouldinIndex.compute(self, cluster_avg_intra_dist,
                                                                    list(cluster_dist._weights))
            self.callback_dbi(self.n_iter_, self.DBI)
            tol1 = self.transfer_variable.guest_tol.get(idx=0, suffix=(self.n_iter_,))
            tol2 = self.transfer_variable.host_tol.get(idx=0, suffix=(self.n_iter_,))
            tol_final = tol1 + tol2
            self.is_converged = True if tol_final < self.tol else False
            self.transfer_variable.arbiter_tol.remote(self.is_converged, role=consts.HOST, idx=-1,
                                                      suffix=(self.n_iter_,))
            self.transfer_variable.arbiter_tol.remote(self.is_converged, role=consts.GUEST, idx=0,
                                                      suffix=(self.n_iter_,))

            self.n_iter_ += 1

            if self.is_converged:
                break

    def predict(self, data_instances=None):
        LOGGER.info("Start predict ...")
        res_dict = self.transfer_variable.guest_dist.get(idx=0, suffix='predict')
        host_dicts = self.transfer_variable.host_dist.get(idx=-1, suffix='predict')
        for host_dict in host_dicts:
            res_dict = res_dict.join(host_dict, lambda v1, v2: v1 + v2)
        cluster_result = res_dict.mapValues(lambda v: np.argmin(v))
        cluster_dist_result = res_dict.mapValues(lambda v: min(v))
        self.transfer_variable.cluster_result.remote(cluster_result, role=consts.GUEST, idx=0, suffix='predict')
        self.transfer_variable.cluster_result.remote(cluster_result, role=consts.HOST, idx=-1, suffix='predict')

        dist_cluster_dtable = res_dict.join(cluster_result, lambda v1, v2: [v1, v2])
        dist_table = self.cal_ave_dist(dist_cluster_dtable, cluster_result, self.k)  # ave dist in each cluster
        cluster_dist = self.cluster_dist_aggregator.sum_model(suffix='predict')

        dist_cluster_dtable_out = cluster_result.join(cluster_dist_result, lambda v1, v2: [int(v1), float(v2)])
        cluster_max_radius = dist_cluster_dtable_out.mapPartitions(self.max_radius).reduce(self.get_max_radius)
        result = []
        for i in range(self.k):
            result.append(
                tuple([i, [dist_table[i][1], dist_table[i][2], cluster_max_radius[i], list(cluster_dist._weights)]]))
        predict_result1 = session.parallelize(result, partition=res_dict.partitions, include_key=True)
        predict_result2 = dist_cluster_dtable_out
        return predict_result1, predict_result2
