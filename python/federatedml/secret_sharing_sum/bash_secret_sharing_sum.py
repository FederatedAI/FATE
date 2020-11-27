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

from python.fate_arch.session import computing_session
from python.federatedml.model_base import ModelBase


class BaseSecretSharingSum(ModelBase):
    def __init__(self):
        super(BaseSecretSharingSum, self).__init__()
        self.x = None
        self.y_recv = []
        self.x_plus_y = None
        self.host_sum_recv = []
        self.prime = None
        self.q = None
        self.g = 2
        self.secret_sharing = []  # (x,f(x))
        self.commitments = []  # (x,g(ai))
        self.share_amount = None
        self.partition = None
        self.coefficients = None
        self.secret_sum = None

    def secure(self):
        self.generate_shares()
        self.generate_commitments()

    def generate_coefficients(self, x):
        coefficients = [x[1]]
        for i in range(self.share_amount-1):
            random_coefficients = np.random.randint(0, self.prime-1)
            coefficients.append(random_coefficients)
        res = (x[0], coefficients)
        return res

    def calculate_polynomial(self, coefficients):
        uid, coefficient = coefficients[0], coefficients[1]
        fx = []
        for x in range(1, self.share_amount+1):
            y = 0
            for index, c in enumerate(coefficient):
                exponentiation = (x ** index) % self.prime
                term = (c * exponentiation) % self.prime
                y = (y + term) % self.prime
            fx.append((x, y))
        res = (uid, fx)
        return res

    def generate_shares(self):
        self.coefficients = list(map(self.generate_coefficients, list(self.x.collect())))
        fx = list(map(self.calculate_polynomial, self.coefficients))
        fx_table = computing_session.parallelize(fx, include_key=True, partition=self.partition)
        for i in range(self.share_amount):
            self.secret_sharing.append(fx_table.mapValues(lambda y: y[i]))

    def generate_commitments(self):
        self.commitments = list(map(self.calculate_commitments, self.coefficients))

    def calculate_commitments(self, coefficients):
        uid, coefficient = coefficients[0], coefficients[1]
        commitment = list(map(lambda c: self.g ** c, coefficient))
        res = (uid, commitment)
        return res

    def sharing_sum(self):
        for recv in self.y_recv:
            self.x_plus_y = self.x_plus_y.union(recv, lambda x, y: (x[0], x[1]+y[1]))

    def reconstruct(self):
        self.x_plus_y = self.x_plus_y.mapValues(lambda x: [x])
        for recv in self.host_sum_recv:
            self.x_plus_y = self.x_plus_y.union(recv, lambda x, y: x+[y])
        free_coefficient = map(self.modular_lagrange_interpolation, list(self.x_plus_y.collect()))
        self.secret_sum = list(free_coefficient)

    def modular_lagrange_interpolation(self, secret_sharing):
        uid, points = secret_sharing[0], secret_sharing[1]
        x_values, y_values = zip(*points)
        f_x = 0
        for i in range(len(points)):
            numerator, denominator = 1, 1
            for j in range(len(points)):
                if i == j:
                    continue
                numerator = (numerator * (0 - x_values[j])) % self.prime
                denominator = (denominator * (x_values[i] - x_values[j])) % self.prime
            lagrange_polynomial = numerator * self.mod_inverse(denominator)
            f_x = (self.prime + f_x + (y_values[i] * lagrange_polynomial)) % self.prime
        res = (uid, f_x)
        return res

    def egcd(self, a, b):
        if a == 0:
            res = (b, 0, 1)
        else:
            g, y, x = self.egcd(b % a, a)
            res = (g, x - (b // a) * y, y)
        return res

    def mod_inverse(self, k):
        k = k % self.prime
        if k < 0:
            r = self.egcd(self.prime, -k)[2]
        else:
            r = self.egcd(self.prime, k)[2]
        res = (self.prime + r) % self.prime
        return res
