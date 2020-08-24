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

import random
import functools
import numpy as np
import math
from arch.api import session
from arch.api.utils import log_utils
from federatedml.unsupervise.kmeans.kmeans_model_base import BaseKmeansModel
from federatedml.param.hetero_kmeans_param import KmeansParam
from federatedml.util import consts
from federatedml.util import fate_operator
from federatedml.framework.homo.blocks import secure_sum_aggregator
from federatedml.framework.weights import NumpyWeights

LOGGER = log_utils.getLogger()


class HeteroKmeansClient(BaseKmeansModel):
    def __init__(self):
        super(HeteroKmeansClient, self).__init__()
        self.dist_aggregator = secure_sum_aggregator.Client(enable_secure_aggregate=True)
        self.cluster_dist_aggregator = secure_sum_aggregator.Client(enable_secure_aggregate=True)
        self.client_dist = None
        self.client_tol = None

    @staticmethod
    def educl_dist(u, centroid_list):
        result = []
        for c in centroid_list:
            result.append(np.sqrt(np.sum(np.power(c - u.features, 2))))
        return result

    def get_centroid(self,data_instances):
        random.seed(self.k)
        random_list = list()
        feature_list = list()
        v_list = list()
        for r in range(0,self.k):
            random_list.append(math.ceil(random.random()*data_instances.count()))
        n = 0
        sorted_k = sorted(list(data_instances.mapValues(lambda data_instance:None).collect()), key = lambda k: k[0])
        for k in sorted_k:
            if n in random_list:
                v_list.append(k[0])
            n += 1
        LOGGER.info(v_list)
        for target in v_list:
            feature_list.append(data_instances.get(target).features)
        LOGGER.info(feature_list)
        return feature_list

    @staticmethod
    def get_sum(x1, x2):
        feature_sum = x1 + x2
        return feature_sum

    def centroid_cal(self, cluster_result, data_instances):
        sorted_k = sorted(list(data_instances.mapValues(lambda data_instance: None).collect()), key=lambda k: k[0])
        cluster_result_dtable = session.parallelize(zip(np.array(sorted_k)[:, 0], cluster_result), include_key=True, partition=data_instances._partitions)
        get_sum = functools.partial(self.get_sum)
        centroid_list = list()
        for kk in range(0, self.k):
            a = cluster_result_dtable.filter(lambda k1, v1: v1 == kk)
            centroid_k = a.join(data_instances, lambda v1, v2: v2).mapValues(lambda v: v.features).reduce(get_sum) / a.count()
            centroid_list.append(centroid_k)
        return centroid_list

    def centroid_dist(self, centroid_list):
        cluster_dist_list =[]
        for i in range(0, len(centroid_list)):
            for j in range(0, len(centroid_list)):
                if j!=i:
                    cluster_dist_list.append(np.sum((np.array(centroid_list[i])-np.array(centroid_list[j]))**2))
        return cluster_dist_list

    def fit(self, data_instances):
        LOGGER.info("Enter hetero_kmenas_client fit")
        self._abnormal_detection(data_instances)
        self.centroid_list = self.get_centroid(data_instances)
        while self.n_iter_ < self.max_iter:
            d = functools.partial(self.educl_dist, centroid_list=self.centroid_list)
            dist_all_dtable = data_instances.mapValues(d)
            sorted_dist_table = sorted(list(dist_all_dtable.collect()), key=lambda k : k[0])
            dist_all = np.array([v[1] for v in sorted_dist_table])
            self.dist_aggregator.send_model(NumpyWeights(dist_all), suffix=(self.n_iter_,))
            cluster_result = self.transfer_variable.cluster_result.get(idx=0, suffix=(self.n_iter_,))
            centroid_new = self.centroid_cal(cluster_result, data_instances)
            client_tol = np.sum(np.sum((np.array(self.centroid_list) - np.array(centroid_new))**2,axis=1))
            self.centroid_list = centroid_new
            self.cluster_result = cluster_result
            self.client_tol.remote(client_tol, role=consts.ARBITER, idx=0, suffix=(self.n_iter_,))
            cluster_dist = self.centroid_dist(self.centroid_list)
            self.cluster_dist_aggregator.send_model(cluster_dist, suffix=(self.n_iter_,))
            tol_tag = self.transfer_variable.arbiter_tol.get(idx=0, suffix=(self.n_iter_,))
            self.n_iter_ += 1
            if tol_tag:
                break

    def predict(self, data_instances):
        LOGGER.info("Start predict ...")
        d = functools.partial(self.educl_dist, centroid_list=self.centroid_list)
        dist_all_dtable = data_instances.mapValues(d)
        sorted_dist_table = sorted(list(dist_all_dtable.collect()), key=lambda k: k[0])
        dist_all = np.array([v[1] for v in sorted_dist_table])
        self.dist_aggregator.send_model(NumpyWeights(dist_all), suffix='predict')
        sample_class = self.transfer_variable.cluster_result.get(idx=0)
        cluster_dist = self.centroid_dist(self.centroid_list)
        self.cluster_dist_aggregator.send_model(cluster_dist, suffix='predict')
        dist_fake = [-1] * data_instances.count()
        sorted_key = list(sorted_dist_table.mapValues(lambda k: k[0]).collect())
        predict_concat = session.parallelize(tuple(zip(sorted_key,sample_class,dist_fake,dist_fake)), partition=data_instances.patitions)
        predict_result = data_instances.join(predict_concat, lambda v1, v2: [v1.label, v2[0], v2[1], v2[2]])
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
