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

import sys

from arch.api import eggroll


# from arch.api.proto.feature_engineer_result_pb2 import FeatureSelectResults


def show_result(table, namespace, rows=10):
    result = eggroll.table(table, namespace)
    print('data count: {}'.format(result.count()))

    if result.count() > 10:
        result_data = result.collect()
        n = 0
        while n < rows:
            result = result_data.__next__()
            print("predict result: {}".format(result[1].features))
            n += 1


# def show_model(table, namespace):
#     model = eggroll.table(table, namespace)
#     model_local = model.collect()
#     serialize_str = model_local.__next__()[1]
#     results = FeatureSelectResults()
#     results.ParseFromString(serialize_str)
#     for r in results.results:
#         print(r)


if __name__ == '__main__':
    table = sys.argv[1]
    namespace = sys.argv[2]
    rows = int(sys.argv[3])
    show_item = sys.argv[4]
    mode = int(sys.argv[5])
    eggroll.init("any_one", mode)

    if show_item == 'predict_data':
        show_result(table, namespace, rows)

