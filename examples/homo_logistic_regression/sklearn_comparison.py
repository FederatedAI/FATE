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

import csv
import os
import time

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

home_dir = os.path.split(os.path.realpath(__file__))[0]

normal_train_data = home_dir + "/../data/mimic_data.csv"
normal_test_data = home_dir + "/../data/mimic_homo_test.csv"


def get_auc(y_true, y_score):
    auc = metrics.roc_auc_score(y_true, y_score)
    print("The auc is :{}".format(auc))


def read_data(input_file='', head=True):
    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file)
        if head is True:
            csv_head = next(csv_reader)

        for row in csv_reader:
            yield (row[0], row[1], row[2:])


def load_data(data_set):
    d_generator = read_data(data_set, True)
    X = []
    Y = []
    for _, y, x in d_generator:
        X.append(x)
        Y.append(y)

    X = np.array(X, 'float')
    Y = np.array(Y, 'int')
    return X, Y


def train():
    train_x, train_y = load_data(normal_train_data)
    test_x, test_y = load_data(normal_test_data)

    model = LogisticRegression(solver='sag', tol=1e-5, max_iter=10)
    t0 = time.time()
    model.fit(train_x, train_y)
    t1 = time.time()
    print("model fit time: {}".format(t1 - t0))
    n_iter = int(model.n_iter_)
    print("time per iter: {}".format((t1 - t0) / n_iter))
    proba_y = model.predict_proba(test_x)[:, 1]
    predict_y = model.predict(test_x)
    print(proba_y)
    # print("The prob of y is:", proba_y)
    print("model n_iter_: {}".format(model.n_iter_))

    get_auc(test_y, proba_y)
    precision = metrics.precision_score(test_y, predict_y)
    print("precision: {}".format(precision))
    print("coef_: {}".format(model.coef_))
    print("intercept_: {}".format(model.intercept_))


if __name__ == '__main__':
    train()
