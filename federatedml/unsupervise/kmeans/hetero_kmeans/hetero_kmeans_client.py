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
import math
import random

import numpy as np

from fate_arch.session import computing_session as session
from federatedml.framework.homo.blocks import secure_sum_aggregator
from federatedml.framework.homo.procedure import table_aggregator
from federatedml.framework.weights import NumpyWeights
from federatedml.unsupervise.kmeans.kmeans_model_base import BaseKmeansModel
from federatedml.util import LOGGER
from federatedml.util import consts


class HeteroKmeansClient(BaseKmeansModel):
    def __init__(self):
        super(HeteroKmeansClient, self).__init__()
        self.dist_aggregator = secure_sum_aggregator.Client(enable_secure_aggregate=True)
        self.cluster_dist_aggregator = secure_sum_aggregator.Client(enable_secure_aggregate=True)
        self.client_dist = None
        self.client_tol = None
        self.aggregator = table_aggregator.Client()

    @staticmethod
    def educl_dist(u, centroid_list):
        result = []
        for c in centroid_list:
            result.append(np.sum(np.power(np.array(c) - u.features, 2)))
        return result

    def get_centroid(self, data_instances):
        random.seed(self.k)
        random_list = list()
        v_list = list()
        for r in range(0, self.k):
            random_list.append(math.ceil(random.random() * data_instances.count()))
        n = 0
        key = list(data_instances.mapValues(lambda data_instance: None).collect())
        for k in key:
            if n in random_list:
                v_list.append(k[0])
            n += 1
        return v_list

    def f(self, iterator):
        cluster_result = dict()
        for k, v in iterator:
            if v[1] not in cluster_result:
                cluster_result[v[1]] = v[0]
            else:
                cluster_result[v[1]] += v[0]
        return cluster_result

    def centroid_cal(self, cluster_result, data_instances):
        cluster_result_dtable = data_instances.join(cluster_result, lambda v1, v2: [v1.features, v2])
        centroid_feature_sum = cluster_result_dtable.mapPartitions(self.f).reduce(self.sum_dict)
        cluster_count = cluster_result.mapPartitions(self.count).reduce(self.sum_dict)
        centroid_list = []
        cluster_count_list = []
        count_all = data_instances.count()
        for k in centroid_feature_sum:
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
        np.random.seed(data_instances.count())
        if self.role == consts.GUEST:
            first_centroid_key = self.get_centroid(data_instances)
            self.transfer_variable.centroid_list.remote(first_centroid_key, role=consts.HOST, idx=-1)
            # rand = np.random.rand(data_instances.count())
        else:
            first_centroid_key = self.transfer_variable.centroid_list.get(idx=0)
            # rand = -np.random.rand(data_instances.count())
        key_dtable = session.parallelize(tuple(zip(first_centroid_key, first_centroid_key)),
                                         partition=data_instances.partitions, include_key=True)
        centroid_list = list(key_dtable.join(data_instances, lambda v1, v2: v2.features).collect())
        self.centroid_list = [v[1] for v in centroid_list]

        while self.n_iter_ < self.max_iter:
            d = functools.partial(self.educl_dist, centroid_list=self.centroid_list)
            dist_all_dtable = data_instances.mapValues(d)

            self.aggregator.send_table(dist_all_dtable, suffix=(self.n_iter_,))
            cluster_result = self.aggregator.get_aggregated_table(suffix=(self.n_iter_,))

            centroid_new, self.cluster_count = self.centroid_cal(cluster_result, data_instances)
            self.centroid_list = centroid_new
            self.cluster_result = cluster_result
            cluster_dist = self.centroid_dist(self.centroid_list)
            self.cluster_dist_aggregator.send_model(NumpyWeights(np.array(cluster_dist)), suffix=(self.n_iter_,))
            client_tol = np.sum(np.sum((np.array(self.centroid_list) - np.array(centroid_new)) ** 2, axis=1))
            self.client_tol.remote(client_tol, role=consts.ARBITER, idx=0, suffix=(self.n_iter_,))
            self.is_converged = self.transfer_variable.arbiter_tol.get(idx=0, suffix=(self.n_iter_,))
            self.n_iter_ += 1

            if self.is_converged:
                break

    def predict(self, data_instances):
        LOGGER.info("Start predict ...")
        np.random.seed(data_instances.count())
        if self.role == consts.GUEST:
            rand = np.random.rand(data_instances.count())
        else:
            rand = -np.random.rand(data_instances.count())
        d = functools.partial(self.educl_dist, centroid_list=self.centroid_list)
        dist_all_dtable = data_instances.mapValues(d)
        sorted_key = sorted(list(dist_all_dtable.mapValues(lambda v: None).collect()), key=lambda k: k[0])
        key = [k[0] for k in sorted_key]
        key_secureagg = session.parallelize(tuple(zip(key, rand)), partition=data_instances.partitions,
                                            include_key=True)
        secure_dist_all = key_secureagg.join(dist_all_dtable, lambda v1, v2: v1 + v2)
        self.client_dist.remote(secure_dist_all, role=consts.ARBITER, idx=0, suffix='predict')
        cluster_result = self.transfer_variable.cluster_result.get(idx=0, suffix='predict')
        cluster_dist = self.centroid_dist(self.centroid_list)
        self.cluster_dist_aggregator.send_model(NumpyWeights(np.array(cluster_dist)), suffix='predict')
        predict_result = data_instances.join(cluster_result, lambda v1, v2: [v1.label, int(v2)])
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
