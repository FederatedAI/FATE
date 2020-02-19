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

    def educlDist(self, x, c):
        return sqrt(sum(power(c-x, 2)))

    def get_centriod(self):
        pass

    def fit(self, data_instances):
        LOGGER.info("Enter hetero_kmenas_guest fit")
        self._abnormal_detection(data_instances)
        #self.header = self.get_header(data_instances)

        n = inf

        while self.n_iter_ < self.max_iter and n > self.tol:
            centriod = self.get_centriod()
            dist_table = mat(zeros(data_instances.shape[0],self.k))
            for i in range(0, self.k):
                d = functools.partial(self.educlDist, c=centriod[i])
                dist = data_instances.mapValue(d)
                dist_r = dist.mapValue(lambda x: x+random.random())
                self.transfer_variable.guest_dist.remote(dist_r, role=consts.ARBITER, idx=-1, suffix=self.n_iter_)
            self.n_iter_ += 1




