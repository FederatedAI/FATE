import math
import unittest

import numpy as np

from arch.api import eggroll
from federatedml.util import consts

eggroll.init("123")

from federatedml.statistic.statics import MultivariateStatisticalSummary
from federatedml.feature.instance import Instance


class TestStatistics(unittest.TestCase):
    def setUp(self):
        self.count = 1000
        self.cols = 10
        self.datas = []

        self.eps = 1e-5
        table = []
        for col in range(self.cols):
            tmp_data = np.random.randn(self.count)
            table.append(tmp_data)
            self.datas.append(tmp_data)

        table = np.array(table)
        table = table.transpose()
        table_data = []
        for i in range(self.count):
            tmp_data = table[i, :]
            tmp = Instance(inst_id=i, features=np.array(tmp_data), label=0)
            table_data.append((i, tmp))

        self.table = eggroll.parallelize(table_data,
                                         include_key=True,
                                         partition=10)
        self.summary_obj = MultivariateStatisticalSummary(self.table, -1)

    def test_MultivariateStatisticalSummary(self):

        for col in range(self.cols):
            this_data = self.datas[col]
            mean = self.summary_obj.get_mean()[col]
            var = self.summary_obj.get_variance()[col]
            max_value = self.summary_obj.get_max()[col]
            min_value = self.summary_obj.get_min()[col]

            real_max = np.max(this_data)
            real_min = np.min(this_data)
            real_mean = np.mean(this_data)
            real_var = np.var(this_data)

            self.assertTrue(math.fabs(mean - real_mean) < self.eps)
            self.assertTrue(math.fabs(var - real_var) < self.eps)
            self.assertTrue(max_value == real_max)
            self.assertTrue(min_value == real_min)

    def test_median(self):
        error = consts.DEFAULT_RELATIVE_ERROR
        medians = self.summary_obj.get_median()

        for idx, this_data in enumerate(self.datas):
            sort_data = sorted(this_data)
            min_rank = int(math.floor((0.5 - 2 * error) * self.count))
            max_rank = int(math.ceil((0.5 + 2 * error) * self.count))
            self.assertTrue(sort_data[min_rank] <= medians[idx] <= sort_data[max_rank])

    def tearDown(self):
        self.table.destroy()


if __name__ == '__main__':
    unittest.main()
