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
from sklearn.datasets import make_moons
from arch.api import session
# from arch.api.cluster import mock_roll as eggroll
import functools
import uuid
import time
from arch.api import WorkMode

current_milli_time = lambda: int(round(time.time() * 1000))


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def _calculate_yA_uA(input, length):
    U_A = input[0]
    y = input[1]
    return np.expand_dims(np.sum(y * U_A, axis=0) / length, axis=0)


def cal_mul(input):
    U_A = input[0]
    y = input[1]
    length = input[2]
    return y * U_A / length


def my_sigmoid(w, b, x):
    return sigmoid(np.dot(w, x) + b)


def add(x, y):
    return x + y


def mul(x, y):
    return x * y


class TestMethod(object):
    def test(self, value):
        return self.mul(len(value))

    def mul(self, x):
        return x * x


if __name__ == '__main__':
    # 修改flow_id 否则内存表可能被覆盖
    session.init(mode=0)
    ns = str(uuid.uuid1())

    X = session.table('testX7', ns, partition=2)
    Y = session.table('testY7', ns, partition=2)

    # X.destroy()
    # Y.destroy()

    b = np.array([0])
    eta = 1.2
    max_iter = 100

    total_num = 500

    _x, _y = make_moons(total_num, noise=0.25)

    for i in range(np.shape(_y)[0]):
        X.put(i, _x[i])
        Y.put(i, _y[i])

    print(len([y for y in Y.collect()]))

    start = current_milli_time()
    shape_w = [1, np.shape(_x)[1]]
    w = np.ones(shape_w)

    # lr = LR(shape_w)
    # lr.train(X, Y)
    itr = 0
    while itr < max_iter:
        H = X.mapValues(functools.partial(my_sigmoid, w, b))
        R = H.join(Y, lambda hx, y: hx - y)
        gradient_w = R.join(X, mul).reduce(add) / total_num
        gradient_b = R.reduce(add) / total_num
        w = w - eta * gradient_w
        b = b - eta * gradient_b
        # self.plot(itr)
        itr += 1

    print("train total time: {}".format(current_milli_time() - start))
    _x_test, _y_test = make_moons(50)

    y_pred = [my_sigmoid(w, b, x)[0] for x in _x_test]
    from sklearn import metrics

    auc = metrics.roc_auc_score(_y_test, y_pred)
    print("auc: {}".format(auc))
