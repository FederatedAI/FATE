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

LOGGER = log_utils.getLogger()




class HeteroKmeansClient(BaseKmeansModel):
    def __init__(self):
        super(HeteroKmeansClient, self).__init__()
        self.client_dist = None
        self.client_tol = None

    @staticmethod
    def educl_dist(x, centroid_list, rand):
        result = []
        for c in centroid_list:
            result.append(sqrt(sum(power(c - x, 2))) + rand)
        return result

    def get_centroid(self):
        pass

    @staticmethod
    def tol_cal(clu1, clu2):
        diffs = 0
        for i in range(0, len(clu1)):
            for j in range(0, len(clu1[1])):
                diffs += power(clu1[i][j] - clu2[i][j], 2)
        return diffs

    @staticmethod
    def get_average(list):
        for items in list:
            sum += items
        return sum / len(list)

    def centroid_cal(self, cluster_result, data_instances, centroids):
        ave = functools.partial(self.get_average)
        for k in range(0,self.k):
            a = cluster_result.filter(lambda k, v : v == k)
            centroid_k = a.union(data_instances, lambda v1, v2 : v2).reduce(ave)
            centroid_list = centroid_list.append(centroid_k)
        return centroid_list

    def fit(self, data_instances, client):
        LOGGER.info("Enter hetero_kmenas_guest fit")
        self._abnormal_detection(data_instances)
        # self.header = self.get_header(data_instances)
        centroids = self.get_centroid()
        while self.n_iter_ < self.max_iter:
            d = functools.partial(self.educl_dist, centroid_list=centroids, rand=random.random())
            dist_all = data_instances.mapValues(d)
            self.client_dist.remote(dist_all, role=consts.ARBITER, idx=0, suffix=self.n_iter_)
            cluster_result= self.transfer_variable.cluster_result.get(idx=0, suffix=self.n_iter_)
            centroid_new = self.centroid_cal(cluster_result, data_instances, centroids)
            guest_tol = self.tol_cal(centroids, centroid_new)
            centroids = centroid_new
            self.client_tol.remote(guest_tol, role=consts.ARBITER, idx=0, suffix=self.n_iter_)
            n = self.transfer_variable.arbiter_tol.get(idx=0, suffix=self.n_iter_)
            if n < self.tol:
                break
            self.n_iter_ += 1


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
