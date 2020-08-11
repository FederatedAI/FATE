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
from numpy import *
from arch.api.utils import log_utils
from federatedml.unsupervise.kmeans.kmeans_model_base import BaseKmeansModel
from federatedml.param.hetero_kmeans_param import KmeansParam
from federatedml.util import consts
from federatedml.util import fate_operator
from federatedml.framework.homo.blocks import secure_sum_aggregator

LOGGER = log_utils.getLogger()


class HeteroKmeansClient(BaseKmeansModel):
    def __init__(self):
        super(HeteroKmeansClient, self).__init__()
        self.dist_aggregator = secure_sum_aggregator.Client(enable_secure_aggregate=True)
        self.client_dist = None
        self.client_tol = None

    @staticmethod
    def educl_dist(u, centroid_list):
        result = []
        for c in centroid_list:
            result.append(sqrt(power(c - u.features, 2)))
        return result

    def get_centroid(self,data_instances):
        random.seed(self.k)
        random_list = list()
        v_list = list()
        for r in range(0,self.k):
            random_list.append(random.random()*data_instances.count())
        n = 0
        for k,v in data_instances.collect():
            if n in random_list:
                v_list.append(v)
            n += 1
        feature_list=list(data_instances.filter(lambda k1 , v1: k1 in v_list).collect())
        return feature_list

    @staticmethod
    def get_sum(x1, x2):
        for items in x1:
            x1.features[items] += x2.features[items]
        return x1.features

    def centroid_cal(self, cluster_result, data_instances, centroids):
        sum_all = functools.partial(self.get_sum)
        centroid_list = list()
        for kk in range(0, self.k):
            a = cluster_result.filter(lambda k1, v1: v1 == kk)
            centroid_k = a.join(data_instances, lambda v1, v2: v2).reduce(sum_all) / a.count
            centroid_list = centroid_list.append(centroid_k)
        return centroid_list

    def fit(self, data_instances):
        LOGGER.info("Enter hetero_kmenas_client fit")
        self._abnormal_detection(data_instances)
        # self.header = self.get_header(data_instances)
        self.centroid_list = self.get_centroid(data_instances)
        tol_sum = inf
        while self.n_iter_ < self.max_iter:
            d = functools.partial(self.educl_dist, centroid_list=self.centroid_list)
            dist_all = data_instances.mapValues(d)
            self.dist_aggregator.send_model(dist_all, suffix=(self.n_iter_,))
            cluster_result = self.transfer_variable.cluster_result.get(idx=0, suffix=(self.n_iter_,))
            centroid_new = self.centroid_cal(cluster_result, data_instances, self.centroid_list)
            client_tol = np.sum((self.centroid_list - centroid_new)**2,axis=1)
            self.centroid_list = centroid_new
            self.cluster_result = cluster_result
            self.client_tol.remote(client_tol, role=consts.ARBITER, idx=0, suffix=(self.n_iter_,))
            tol_tag = self.transfer_variable.arbiter_tol.get(idx=0, suffix=(self.n_iter_,))
            self.n_iter_ += 1
            if tol_tag == 1:
                break

    def predict(self, data_instances):
        LOGGER.info("Start predict ...")
        dist_list = self.educl_dist(data_instances.features, self.centroid_list)
        self.client_dist.remote(dist_list, role=consts.ARBITER, idx=0)
        sample_class = self.transfer_variable.cluster_result.get(idx=0)
        predict_result = data_instances.join(sample_class, lambda d, pred: [d.label, pred, pred, {"label": pred}])
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
