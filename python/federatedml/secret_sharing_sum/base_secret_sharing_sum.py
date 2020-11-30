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

from federatedml.model_base import ModelBase
from federatedml.secureprotol.secretsharing.vss import Vss


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
        fx_table = self.x.mapValues(self.vss.encrypt)
        for i in range(self.share_amount):
            self.secret_sharing.append(fx_table.mapValues(lambda y: y[i]))

    def sharing_sum(self):
        for recv in self.y_recv:
            self.x_plus_y = self.x_plus_y.union(recv, lambda x, y: (x[0], x[1]+y[1]))

    def reconstruct(self):
        self.x_plus_y = self.x_plus_y.mapValues(lambda x: [x])
        for recv in self.host_sum_recv:
            self.x_plus_y = self.x_plus_y.union(recv, lambda x, y: x+[y])
        self.secret_sum = self.x_plus_y.mapValues(self.vss.decrypt)
