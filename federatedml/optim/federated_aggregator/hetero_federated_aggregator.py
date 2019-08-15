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
from arch.api.utils import log_utils
from federatedml.util import fate_operator

LOGGER = log_utils.getLogger()


class HeteroFederatedAggregator(object):
    @staticmethod
    def aggregate_add(table_a, table_b):
        """
        Compute a + b
        Parameters
        ----------
        table_a: DTable, input data a
        table_b: DTable, input data b
        Returns
        ----------
        DTable
            sum of each element in table_a and table_b
        """
        table_add = table_a.join(table_b, lambda a, b: a + b)
        return table_add

    # do res = (a + b)^2
    @staticmethod
    def aggregate_add_square(table_a, table_b, table_a_square, table_b_square):
        """
        Compute （a + b)^2
        Parameters
        ----------
        table_a: DTable, input data a
        table_b: DTable, input data b
        table_a_square: DTable, a^2
        table_b_square: DTable, b^2
        Returns
        ----------
        DTable
            return (a + b)^2
        """
        table_a_mul_b = table_a.join(table_b, lambda a, b: 2 * a * b)
        table_a_square_add_b_square = HeteroFederatedAggregator.aggregate_add(table_a_square, table_b_square)
        table_add_square = HeteroFederatedAggregator.aggregate_add(table_a_mul_b, table_a_square_add_b_square)
        return table_add_square

    @staticmethod
    def separate(value, size_list):
        """
        Separate value in order to several set according size_list
        Parameters
        ----------
        value: list or ndarray, input data
        size_list: list, each set size

        Returns
        ----------
        list
            set after separate
        """
        separate_res = []
        cur = 0
        for size in size_list:
            separate_res.append(value[cur:cur + size])
            cur += size

        return separate_res

    @staticmethod
    def aggregate_mean(table):
        """
        Compute the mean of values in table
        Parameters
        ----------
        table: DTable, input data

        Returns
        ----------
        float or ndarray
            the mean of values in table
        """
        count = table.count()
        reduce_res = table.reduce(fate_operator.reduce_add)
        if isinstance(reduce_res, list):
            reduce_res = np.array(reduce_res)
        reduce_res = reduce_res / count
        return reduce_res
