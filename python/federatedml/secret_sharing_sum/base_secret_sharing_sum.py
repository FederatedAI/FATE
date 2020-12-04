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
#
import numpy as np

from federatedml.model_base import ModelBase
from federatedml.secureprotol.secret_sharing.vss import Vss


class BaseSecretSharingSum(ModelBase):
    def __init__(self):
        super(BaseSecretSharingSum, self).__init__()
        self.x = None
        self.y_recv = []
        self.x_plus_y = None
        self.host_sum_recv = []
        self.vss = Vss()
        self.g = 2
        self.secret_sharing = []  # (x,f(x))
        self.commitments = []  # (x,g(ai))
        self.share_amount = None
        self.need_verify = None
        self.partition = None
        self.coefficients = None
        self.secret_sum = None

    def secure(self):
        self.generate_shares()

    def generate_shares(self):
        fx_table = self.x.mapValues(self.split_secret)
        for i in range(self.share_amount):
            share = fx_table.mapValues(lambda y: np.array(y)[:, i])
            self.secret_sharing.append(share)

    def split_secret(self, values):
        secrets = values.features
        shares = []
        for s in secrets:
            shares.append(self.vss.encrypt(s))
        return shares

    def sharing_sum(self):
        for recv in self.y_recv:
            self.x_plus_y = self.x_plus_y.union(recv, lambda x, y: np.column_stack((x[:, 0], np.add(x[:, 1], y[:, 1]))))

    def reconstruct(self):
        for recv in self.host_sum_recv:
            self.x_plus_y = self.x_plus_y.union(recv, lambda x, y: np.column_stack((x, y)))
        self.secret_sum = self.x_plus_y.mapValues(self.combine)

    def combine(self, values):
        secret_sum = []
        for v in values:
            x_values = v[::2]
            y_values = v[1::2]
            secret_sum.append(self.vss.decrypt(x_values, y_values))
        return secret_sum


