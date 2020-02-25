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


class HeteroKmeansGuest(BaseKmeansModel):
    def __init__(self):
        super(HeteroKmeansGuest, self).__init__()

    @staticmethod
    def educl_dist(x, centriod_list, rand):
        result = []
        for c in centriod_list:
            result.append(sqrt(sum(power(c - x, 2))) + rand)
        return result

    def get_centroid(self):
        pass

    def tol_cal(self, clu1, clu2):
        return diff

    def fit(self, data_instances):
        LOGGER.info("Enter hetero_kmenas_guest fit")
        self._abnormal_detection(data_instances)
        # self.header = self.get_header(data_instances)
        centroids = self.get_centriod()
        while self.n_iter_ < self.max_iter:
            d = functools.partial(self.educl_dist, centriod_list=centroids, rand=random.random())
            dist_all = data_instances.mapValue(d)
            self.transfer_variable.guest_dist.remote(dist_all, role=consts.ARBITER, idx=-1, suffix=self.n_iter_)
            centriod_new = self.transfer_variable.cluster_result.get(idx=-1, suffix=self.n_iter_)
            guest_tol = self.tol_cal(centroids, centriod_new)
            centroids = centriod_new
            self.transfer_variable.guest_tol.remote(guest_tol, role=consts.ARBITER, idx=-1, suffix=self.n_iter_)
            n = self.transfer_variable.arbiter_tol.get(idx=-1, suffix=self.n_iter_)
            if n < self.tol:
                break
            self.n_iter_ += 1
