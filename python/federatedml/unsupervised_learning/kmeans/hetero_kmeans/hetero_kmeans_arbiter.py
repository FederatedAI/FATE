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
from federatedml.model_base import Metric
from federatedml.model_base import MetricMeta
from federatedml.evaluation.metrics import clustering_metric
from federatedml.feature.instance import Instance
from federatedml.framework.hetero.procedure import table_aggregator
from federatedml.param.hetero_kmeans_param import KmeansParam
from federatedml.unsupervised_learning.kmeans.kmeans_model_base import BaseKmeansModel

# Do not delete this import
from federatedml.framework.weights import NumpyWeights

from federatedml.util import LOGGER
from federatedml.util import consts


class HeteroKmeansArbiter(BaseKmeansModel):
    def __init__(self):
        super(HeteroKmeansArbiter, self).__init__()
        self.model_param = KmeansParam()
        # self.dist_aggregator = secure_sum_aggregator.Server(enable_secure_aggregate=False)
        # self.cluster_dist_aggregator = secure_sum_aggregator.Server(enable_secure_aggregate=False)
        self.DBI = 0
        self.aggregator = table_aggregator.Server(enable_secure_aggregate=True)

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

    def cal_ave_dist(self, dist_cluster_table, cluster_result):
        dist_centroid_dist_table = dist_cluster_table.applyPartitions(self.sum_in_cluster).reduce(self.sum_dict)
        cluster_count = cluster_result.applyPartitions(self.count).reduce(self.sum_dict)
        cal_ave_dist_list = []
        for key in cluster_count.keys():
            count = cluster_count[key]
            cal_ave_dist_list.append([key, count, dist_centroid_dist_table[key] / count])
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
        for k1 in v1.keys() | v2.keys():
            rs[k1] = max(v1.get(k1, 0), v2.get(k1, 0))
        return rs

    def cal_dbi(self, dist_sum, cluster_result, suffix):
        dist_cluster_table = dist_sum.join(cluster_result, lambda v1, v2: [v1, v2])
        dist_table = self.cal_ave_dist(dist_cluster_table, cluster_result)  # ave dist in each cluster
        if len(dist_table) == 1:
            raise ValueError('Only one class detected. DBI calculation error')
        cluster_dist = self.aggregator.sum_model(suffix=(suffix,))
        cluster_avg_intra_dist = []
        for i in range(len(dist_table)):
            cluster_avg_intra_dist.append(dist_table[i][2])
        self.DBI = clustering_metric.DaviesBouldinIndex.compute(self, cluster_avg_intra_dist,
                                                                list(cluster_dist._weights))
        self.callback_dbi(suffix - 1, self.DBI)

    def fit(self, data_instances=None, validate_data=None):
        LOGGER.info("Enter hetero Kmeans arbiter fit")
        last_cluster_result = None
        while self.n_iter_ < self.max_iter:
            dist_sum = self.aggregator.aggregate_tables(suffix=(self.n_iter_,))
            if last_cluster_result is not None:
                self.cal_dbi(dist_sum, last_cluster_result, self.n_iter_)
            cluster_result = dist_sum.mapValues(lambda v: np.argmin(v))
            self.aggregator.send_aggregated_tables(cluster_result, suffix=(self.n_iter_,))
            tol1 = self.transfer_variable.guest_tol.get(idx=0, suffix=(self.n_iter_,))
            tol2 = self.transfer_variable.host_tol.get(idx=0, suffix=(self.n_iter_,))
            tol_final = tol1 + tol2
            self.is_converged = True if tol_final < self.tol else False
            LOGGER.debug(f"iter: {self.n_iter_}, tol_final: {tol_final}, tol: {self.tol},"
                         f" is_converge: {self.is_converged}")
            self.transfer_variable.arbiter_tol.remote(self.is_converged, role=consts.HOST, idx=-1,
                                                      suffix=(self.n_iter_,))
            self.transfer_variable.arbiter_tol.remote(self.is_converged, role=consts.GUEST, idx=0,
                                                      suffix=(self.n_iter_,))
            last_cluster_result = cluster_result
            self.n_iter_ += 1

            if self.is_converged:
                break

        # calculate finall round dbi
        dist_sum = self.aggregator.aggregate_tables(suffix=(self.n_iter_,))
        cluster_result = dist_sum.mapValues(lambda v: np.argmin(v))
        self.aggregator.send_aggregated_tables(cluster_result, suffix=(self.n_iter_,))
        self.cal_dbi(dist_sum, last_cluster_result, self.n_iter_)
        dist_sum_dbi = self.aggregator.aggregate_tables(suffix=(self.n_iter_ + 1,))
        self.aggregator.send_aggregated_tables(cluster_result, suffix=(self.n_iter_ + 1,))
        self.cal_dbi(dist_sum_dbi, cluster_result, self.n_iter_ + 1)

    def predict(self, data_instances=None):
        LOGGER.info("Start predict ...")
        res_dict = self.aggregator.aggregate_tables(suffix='predict')
        cluster_result = res_dict.mapValues(lambda v: np.argmin(v))
        cluster_dist_result = res_dict.mapValues(lambda v: min(v))
        self.aggregator.send_aggregated_tables(cluster_result, suffix='predict')

        res_dict_dbi = self.aggregator.aggregate_tables(suffix='predict_dbi')
        self.aggregator.send_aggregated_tables(cluster_result, suffix='predict_dbi')
        dist_cluster_table = res_dict.join(cluster_result, lambda v1, v2: [v1, v2])
        dist_cluster_table_dbi = res_dict_dbi.join(cluster_result, lambda v1, v2: [v1, v2])
        dist_table = self.cal_ave_dist(dist_cluster_table, cluster_result)  # ave dist in each cluster
        dist_table_dbi = self.cal_ave_dist(dist_cluster_table_dbi, cluster_result)
        # if len(dist_table) == 1:
        #    raise ValueError('Only one class detected. DBI calculation error')
        cluster_dist = self.aggregator.sum_model(suffix='predict')

        dist_cluster_table_out = cluster_result.join(cluster_dist_result, lambda v1, v2: [int(v1), float(v2)])
        cluster_max_radius = dist_cluster_table_out.applyPartitions(self.max_radius).reduce(self.get_max_radius)
        result = []
        for i in range(len(dist_table)):
            c_key = dist_table[i][0]
            result.append(tuple(
                [int(c_key),
                 Instance(features=[dist_table[i][1], dist_table_dbi[i][2],
                                    cluster_max_radius[c_key], list(cluster_dist._weights)])]))
        predict_result1 = session.parallelize(result, partition=res_dict.partitions, include_key=True)
        predict_result2 = dist_cluster_table_out.mapValues(lambda x: Instance(features=x))
        return predict_result1, predict_result2
