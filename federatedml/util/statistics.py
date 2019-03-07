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
import functools


class Statistics(object):
    def partition_statistics(self, kvs, method="mean"):
        data = []
        for key, value in kvs:
            data.append(value)

        if method == "mean":
            return np.sum(data), len(data)
        elif method == "median":
            return np.median(data)

    def mean(self, x):
        if type(x).__name__ == "ndarray" or type(x).__name__ == "list":
            return np.mean(x)
        elif type(x).__name__ == "_DTable":
            partition_cal = functools.partial(self.partition_statistics, method="mean")
            sum, num = x.mapPartitions(partition_cal).reduce(
                lambda sum_num1, sum_num2: (sum_num1[0] + sum_num2[0], sum_num1[1] + sum_num2[1]))
            return 1.0 * sum / num
        else:
            raise TypeError("type {} not supported in mean statistics".format(type(x).__name__))

    def median(self, x):
        if type(x).__name__ == "ndarray" or type(x).__name__ == "list":
            return np.median(x)
        elif type(x).__name__ == "_DTable":
            partition_cal = functools.partial(self.partition_statistics, method="median")
            medians = [median for key, median in list(x.mapPartitions(partition_cal).collect())]
            return np.median(medians)
