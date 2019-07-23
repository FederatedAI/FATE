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


def assert_matrix(X, Y):
    x_shape_len = len(X.shape)
    y_shape_len = len(Y.shape)
    assert x_shape_len == y_shape_len

    for index in range(len(X.shape)):
        assert X.shape[index] == Y.shape[index]

    X = X.reshape(-1)
    Y = Y.reshape(-1)
    dim_num = X.shape[0]
    for index in range(dim_num):
        assert round(X[index], 6) == round(Y[index], 6)


def assert_array(X, Y):
    assert X.shape[0] == X.shape[0]

    elem_num = X.shape[0]
    for index in range(elem_num):
        assert round(X[index], 6) == round(Y[index], 6)
