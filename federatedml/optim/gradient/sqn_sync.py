#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import numpy as np

from arch.api.utils import log_utils
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class SqnSyncBase(object):
    def __init__(self):
        self.batch_data_index_transfer = None
        self.host_forwards_transfer = None
        self.forward_hess = None
        self.forward_hess_transfer = None


class Guest(SqnSyncBase):
    def __init__(self):
        super().__init__()
        self.guest_hess_vector = None

    def register_transfer_variable(self, transfer_variable):
        self.batch_data_index_transfer = transfer_variable.sqn_sample_index
        self.guest_hess_vector = transfer_variable.guest_hess_vector
        self.host_forwards_transfer = transfer_variable.host_sqn_forwards
        self.forward_hess_transfer = transfer_variable.forward_hess

    def sync_sample_data(self, data_instances, sample_size, random_seed, suffix=tuple()):
        n = data_instances.count()
        if sample_size >= n:
            sample_rate = 1.0
        else:
            sample_rate = sample_size / n
        sampled_data = data_instances.sample(sample_rate, random_seed)

        batch_index = sampled_data.mapValues(lambda x: None)
        self.batch_data_index_transfer.remote(obj=batch_index,
                                              role=consts.HOST,
                                              suffix=suffix)
        return sampled_data

    def get_host_forwards(self, suffix=tuple()):
        host_forwards = self.host_forwards_transfer.get(idx=-1,
                                                        suffix=suffix)
        return host_forwards

    def remote_forward_hess(self, forward_hess, suffix=tuple()):
        self.forward_hess_transfer.remote(obj=forward_hess,
                                          role=consts.HOST,
                                          suffix=suffix)

    def sync_hess_vector(self, hess_vector, suffix):
        self.guest_hess_vector.remote(obj=hess_vector,
                                      role=consts.ARBITER,
                                      suffix=suffix)


class Host(SqnSyncBase):
    def __init__(self):
        super().__init__()
        self.host_hess_vector = None

    def register_transfer_variable(self, transfer_variable):
        self.batch_data_index_transfer = transfer_variable.sqn_sample_index
        self.host_forwards_transfer = transfer_variable.host_sqn_forwards
        self.host_hess_vector = transfer_variable.host_hess_vector
        self.forward_hess_transfer = transfer_variable.forward_hess

    def sync_sample_data(self, data_instances, suffix=tuple()):
        batch_index = self.batch_data_index_transfer.get(idx=0,
                                                         suffix=suffix)
        sample_data = data_instances.join(batch_index, lambda x, y: x)
        return sample_data

    def remote_host_forwards(self, host_forwards, suffix=tuple()):
        self.host_forwards_transfer.remote(obj=host_forwards,
                                           role=consts.GUEST,
                                           suffix=suffix)

    def get_forward_hess(self, suffix=tuple()):
        forward_hess = self.forward_hess_transfer.get(idx=0,
                                                      suffix=suffix)
        return forward_hess

    def sync_hess_vector(self, hess_vector, suffix):
        self.host_hess_vector.remote(obj=hess_vector,
                                     role=consts.ARBITER,
                                     suffix=suffix)


class Arbiter(object):
    def __init__(self):
        super().__init__()
        self.guest_hess_vector = None
        self.host_hess_vector = None

    def register_transfer_variable(self, transfer_variable):
        self.guest_hess_vector = transfer_variable.guest_hess_vector
        self.host_hess_vector = transfer_variable.host_hess_vector

    def sync_hess_vector(self, suffix):
        guest_hess_vector = self.guest_hess_vector.get(idx=0,
                                                       suffix=suffix)
        host_hess_vectors = self.host_hess_vector.get(idx=-1,
                                                      suffix=suffix)
        host_hess_vectors = [x.reshape(-1) for x in host_hess_vectors]
        hess_vectors = np.hstack((h for h in host_hess_vectors))
        hess_vectors = np.hstack((hess_vectors, guest_hess_vector))
        return hess_vectors
