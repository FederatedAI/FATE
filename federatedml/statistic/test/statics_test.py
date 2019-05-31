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
        self.count = 10
        self.cols = [('x' + str(i)) for i in range(10)]
        self.datas = {}

        self.eps = 1e-5
        table = []
        for col_name in self.cols:
            tmp_data = np.random.randn(self.count)
            table.append(list(tmp_data))
            self.datas[col_name] = tmp_data

        for t in table:
            t.insert(3, None)
        table = np.array(table)
        table = table.transpose()
        table_data = []
        table_data2 = []
        for i in range(self.count + 1):
            tmp_data = table[i, :]
            tmp = Instance(inst_id=i, features=np.array(tmp_data), label=0)
            tmp2 = np.array(tmp_data)
            table_data.append((i, tmp))
            table_data2.append((i, tmp2))

        self.table = eggroll.parallelize(table_data,
                                         include_key=True,
                                         partition=10)

        self.table2 = eggroll.parallelize(table_data2,
                                          include_key=True,
                                          partition=10)

        self.table.schema = {'header': self.cols}
        self.table2.schema = {'header': self.cols}

        self.detect_cols = ['x0', 'x1']
        self.summary_obj = MultivariateStatisticalSummary(self.table, self.detect_cols, abnormal_list=[None])
        self.summary_obj2 = MultivariateStatisticalSummary(self.table2, self.detect_cols, abnormal_list=[None])


    def test_MultivariateStatisticalSummary(self):

        for col in self.detect_cols:
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

        for col_name, this_data in self.datas.items():
            if col_name not in self.detect_cols:
                continue
            sort_data = sorted(this_data)
            min_rank = int(math.floor((0.5 - 2 * error) * self.count))
            max_rank = int(math.ceil((0.5 + 2 * error) * self.count))
            self.assertTrue(sort_data[min_rank] <= medians[col_name] <= sort_data[max_rank])

    def test_MultivariateStatisticalSummary2(self):

        for col in self.detect_cols:
            this_data = self.datas[col]
            mean = self.summary_obj2.get_mean()[col]
            var = self.summary_obj2.get_variance()[col]
            max_value = self.summary_obj2.get_max()[col]
            min_value = self.summary_obj2.get_min()[col]

            real_max = np.max(this_data)
            real_min = np.min(this_data)
            real_mean = np.mean(this_data)
            real_var = np.var(this_data)

            self.assertTrue(math.fabs(mean - real_mean) < self.eps)
            self.assertTrue(math.fabs(var - real_var) < self.eps)
            self.assertTrue(max_value == real_max)
            self.assertTrue(min_value == real_min)

    def test_median2(self):
        error = consts.DEFAULT_RELATIVE_ERROR
        medians = self.summary_obj2.get_median()

        for col_name, this_data in self.datas.items():
            if col_name not in self.detect_cols:
                continue
            sort_data = sorted(this_data)
            min_rank = int(math.floor((0.5 - 2 * error) * self.count))
            max_rank = int(math.ceil((0.5 + 2 * error) * self.count))
            self.assertTrue(sort_data[min_rank] <= medians[col_name] <= sort_data[max_rank])

    def tearDown(self):
        self.table.destroy()


if __name__ == '__main__':
    unittest.main()
