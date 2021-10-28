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

import functools

import numpy as np

from fate_arch.session import computing_session as session
from federatedml.feature.instance import Instance
from federatedml.framework.hetero.procedure import table_aggregator
from federatedml.framework.weights import NumpyWeights
from federatedml.unsupervised_learning.kmeans.kmeans_model_base import BaseKmeansModel
from federatedml.util import LOGGER
from federatedml.util import consts


class HeteroKmeansClient(BaseKmeansModel):
    def __init__(self):
        super(HeteroKmeansClient, self).__init__()
        # self.dist_aggregator = secure_sum_aggregator.Client(enable_secure_aggregate=False)
        # self.cluster_dist_aggregator = secure_sum_aggregator.Client(enable_secure_aggregate=False)
        self.client_dist = None
        self.client_tol = None
        self.aggregator = table_aggregator.Client(enable_secure_aggregate=True)

    @staticmethod
    def educl_dist(u, centroid_list):
        result = []
        for c in centroid_list:
            result.append(np.sum(np.square(np.array(c) - u.features)))
        return np.array(result)

    def get_centroid(self, data_instances):
        random_key = []
        key = list(data_instances.mapValues(lambda data_instance: None).collect())
        random_list = list(np.random.choice(data_instances.count(), self.k, replace=False))
        for k in random_list:
            random_key.append(key[k][0])
        return random_key

    def cluster_sum(self, iterator):
        cluster_result = dict()
        for k, v in iterator:
            if v[1] not in cluster_result:
                cluster_result[v[1]] = v[0]
            else:
                cluster_result[v[1]] += v[0]
        return cluster_result

    def centroid_cal(self, cluster_result, data_instances):
        cluster_result_table = data_instances.join(cluster_result, lambda v1, v2: [v1.features, v2])
        centroid_feature_sum = cluster_result_table.applyPartitions(self.cluster_sum).reduce(self.sum_dict)
        cluster_count = cluster_result.applyPartitions(self.count).reduce(self.sum_dict)
        centroid_list = []
        cluster_count_list = []
        count_all = data_instances.count()
        # for k in centroid_feature_sum:
        for k in range(self.k):
            if k not in centroid_feature_sum:
                centroid_list.append(self.centroid_list[int(k)])
                cluster_count_list.append([k, 0, 0])
            else:
                count = cluster_count[k]
                centroid_list.append(centroid_feature_sum[k] / count)
                cluster_count_list.append([k, count, count / count_all])
        return centroid_list, cluster_count_list

    def centroid_dist(self, centroid_list):
        cluster_dist_list = []
        for i in range(0, len(centroid_list)):
            for j in range(0, len(centroid_list)):
                if j != i:
                    cluster_dist_list.append(np.sum((np.array(centroid_list[i]) - np.array(centroid_list[j])) ** 2))
        return cluster_dist_list

    def fit(self, data_instances, validate_data=None):
        LOGGER.info("Enter hetero_kmenas_client fit")
        self.header = self.get_header(data_instances)
        self._abnormal_detection(data_instances)
        if self.k > data_instances.count() or self.k < 2:
            raise ValueError('K is too larger or too samll for current data')

        # Get initialized centroid
        np.random.seed(self.random_stat)
        if self.role == consts.GUEST:
            first_centroid_key = self.get_centroid(data_instances)
            self.transfer_variable.centroid_list.remote(first_centroid_key, role=consts.HOST, idx=-1)
            # rand = np.random.rand(data_instances.count())
        else:
            first_centroid_key = self.transfer_variable.centroid_list.get(idx=0)
            # rand = -np.random.rand(data_instances.count())
        key_table = session.parallelize(tuple(zip(first_centroid_key, first_centroid_key)),
                                        partition=data_instances.partitions, include_key=True)
        centroid_list = list(key_table.join(data_instances, lambda v1, v2: v2.features).collect())
        self.centroid_list = [v[1] for v in centroid_list]

        while self.n_iter_ < self.max_iter:
            self.send_cluster_dist(self.n_iter_, self.centroid_list)
            d = functools.partial(self.educl_dist, centroid_list=self.centroid_list)
            dist_all_table = data_instances.mapValues(d)
            cluster_result = self.aggregator.aggregate_then_get_table(dist_all_table, suffix=(self.n_iter_,))
            centroid_new, self.cluster_count = self.centroid_cal(cluster_result, data_instances)

            # cluster_dist = self.centroid_dist(self.centroid_list)
            # self.cluster_dist_aggregator.send_model(NumpyWeights(np.array(cluster_dist)), suffix=(self.n_iter_,))
            client_tol = np.sum(np.sum((np.array(self.centroid_list) - np.array(centroid_new)) ** 2, axis=1))
            self.client_tol.remote(client_tol, role=consts.ARBITER, idx=0, suffix=(self.n_iter_,))
            self.is_converged = self.transfer_variable.arbiter_tol.get(idx=0, suffix=(self.n_iter_,))

            self.centroid_list = centroid_new
            self.cluster_result = cluster_result

            self.n_iter_ += 1

            LOGGER.info(f"iter: {self.n_iter_}, is_converged: {self.is_converged}")

            if self.is_converged:
                break

        # calculate finall round dbi
        self.extra_dbi(data_instances, self.n_iter_, self.centroid_list)
        centroid_new, self.cluster_count = self.centroid_cal(self.cluster_result, data_instances)
        self.extra_dbi(data_instances, (self.n_iter_ + 1), centroid_new)
        # LOGGER.debug(f"Final centroid list: {self.centroid_list}")

    def extra_dbi(self, data_instances, suffix, centroids):
        d = functools.partial(self.educl_dist, centroid_list=centroids)
        dist_all_table = data_instances.mapValues(d)
        self.cluster_result = self.aggregator.aggregate_then_get_table(dist_all_table, suffix=(suffix,))
        self.send_cluster_dist(suffix, centroids)

    def send_cluster_dist(self, suffix, centroids):
        cluster_dist = self.centroid_dist(centroids)
        self.aggregator.send_model(NumpyWeights(np.array(cluster_dist)), suffix=(suffix,))

    def predict(self, data_instances):
        LOGGER.info("Start predict ...")
        self.header = self.get_header(data_instances)
        self._abnormal_detection(data_instances)
        d = functools.partial(self.educl_dist, centroid_list=self.centroid_list)
        dist_all_table = data_instances.mapValues(d)
        cluster_result = self.aggregator.aggregate_then_get_table(dist_all_table, suffix='predict')
        centroid_new, self.cluster_count = self.centroid_cal(cluster_result, data_instances)
        d = functools.partial(self.educl_dist, centroid_list=centroid_new)
        dist_all_table = data_instances.mapValues(d)
        cluster_result_dbi = self.aggregator.aggregate_then_get_table(dist_all_table, suffix='predict_dbi')
        cluster_dist = self.centroid_dist(centroid_new)
        self.aggregator.send_model(NumpyWeights(np.array(cluster_dist)), suffix='predict')
        LOGGER.debug(f"first_data: {data_instances.first()[1].__dict__}")
        predict_result = data_instances.join(cluster_result, lambda v1, v2: Instance(
            features=[v1.label, int(v2)], inst_id=v1.inst_id))
        LOGGER.debug(f"predict_data: {predict_result.first()[1].__dict__}")

        return predict_result


class HeteroKmeansGuest(HeteroKmeansClient):
    def __init__(self):
        super(HeteroKmeansGuest, self).__init__()
        self.client_dist = self.transfer_variable.guest_dist
        self.client_tol = self.transfer_variable.guest_tol


class HeteroKmeansHost(HeteroKmeansClient):
    def __init__(self):
        super(HeteroKmeansHost, self).__init__()
        self.client_dist = self.transfer_variable.host_dist
        self.client_tol = self.transfer_variable.host_tol
