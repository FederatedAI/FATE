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
from federatedml.param.feldman_verifiable_sum_param import FeldmanVerifiableSumParam
from federatedml.secureprotol.secret_sharing.verifiable_secret_sharing.feldman_verifiable_secret_sharing \
    import FeldmanVerifiableSecretSharing


class BaseFeldmanVerifiableSum(ModelBase):
    def __init__(self):
        super(BaseFeldmanVerifiableSum, self).__init__()
        self.vss = FeldmanVerifiableSecretSharing()
        self.host_count = None
        self.model_param = FeldmanVerifiableSumParam()
        self.sum_cols = None
        self.x = None
        self.sub_key = []  # (x,f(x))
        self.commitments = None  # (x,g(ai))
        self.y_recv = []
        self.commitments_recv = []
        self.host_sum_recv = []
        self.x_plus_y = None
        self.secret_sum = None

    def secure(self):
        encrypt_result = self.x.mapValues(self.generate_shares)
        sub_key_table = encrypt_result.mapValues(lambda x: x[0])
        self.commitments = encrypt_result.mapValues(lambda x: x[1])
        for i in range(self.host_count + 1):
            sub_key = sub_key_table.mapValues(lambda y: y[:, i])
            self.sub_key.append(sub_key)

    def generate_shares(self, values):
        keys = []
        commitments = []
        for s in values:
            sub_key, commitment = self.vss.encrypt(s)
            keys.append(sub_key)
            commitments.append(commitment)
        res = (np.array(keys), np.array(commitments))
        return res

    def sub_key_sum(self):
        for recv in self.y_recv:
            self.x_plus_y = self.x_plus_y.join(recv, lambda x, y: np.column_stack((x[:, 0], np.add(x[:, 1], y[:, 1]))))

    def reconstruct(self):
        for recv in self.host_sum_recv:
            self.x_plus_y = self.x_plus_y.join(recv, lambda x, y: np.column_stack((x, y)))
        self.secret_sum = self.x_plus_y.mapValues(self.decrypt)

    def decrypt(self, values):
        secret_sum = []
        for v in values:
            x_values = v[::2]
            y_values = v[1::2]
            secret_sum.append(self.vss.decrypt(x_values, y_values))
        return secret_sum

    def verify_sumkey(self, sum_key, commitment, party_id):
        for recv in self.commitments_recv:
            commitment = commitment.join(recv, lambda x, y: (x * y) % self.vss.p)
        sum_key.join(commitment, lambda x, y: self.verify(x, y, party_id, "sum_key"))

    def verify_subkey(self, sub_key, commitment, party_id):
        sub_key.join(commitment, lambda x, y: self.verify(x, y, party_id, "sub_key"))

    def verify(self, key, commitment, party_id, key_type):
        for idx, key in enumerate(key):
            res = self.vss.verify(key, commitment[idx])
            if not res:
                raise ValueError(f"Get wrong {key_type} from {party_id}")
        return True
