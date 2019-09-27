import math
import unittest

import numpy as np

from arch.api import session
from federatedml.util import consts

session.init("123")

from federatedml.statistic.statics import MultivariateStatisticalSummary
from federatedml.feature.instance import Instance


class TestStatistics(unittest.TestCase):
    def setUp(self):
        session.init("test_instance")
        dense_inst = []
        dense_not_inst = []
        headers = ['x' + str(i) for i in range(20)]
        self.header = headers
        self.eps = 1e-5
        self.count = 100
        self.dense_data_transpose = []
        for i in range(self.count):
            features = i % 16 * np.ones(20)
            inst = Instance(features=features)
            dense_inst.append((i, inst))
            self.dense_data_transpose.append(features)
            dense_not_inst.append((i, features))
        self.dense_inst = dense_inst
        self.dense_not_inst = dense_not_inst
        self.dense_data_transpose = np.array(self.dense_data_transpose)
        self.dense_data_transpose = self.dense_data_transpose.transpose()

        self.dense_table = session.parallelize(dense_inst, include_key=True, partition=5)
        self.dense_not_inst_table = session.parallelize(dense_not_inst, include_key=True, partition=5)
        self.dense_table.schema = {'header': headers}
        self.dense_not_inst_table.schema = {'header': headers}

        col_index = [1, 2, 3]
        self.col_index = col_index
        self.summary_obj = MultivariateStatisticalSummary(self.dense_table, col_index, abnormal_list=[None])
        self.summary_obj_not_inst = MultivariateStatisticalSummary(self.dense_not_inst_table, col_index,
                                                                   abnormal_list=[None])

    def test_MultivariateStatisticalSummary(self):

        for col in self.col_index:
            col_name = self.header[col]
            this_data = self.dense_data_transpose[col]
            mean = self.summary_obj.get_mean()[col_name]
            var = self.summary_obj.get_variance()[col_name]
            max_value = self.summary_obj.get_max()[col_name]
            min_value = self.summary_obj.get_min()[col_name]

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

        for col_idx in self.col_index:
            col_name = self.header[col_idx]
            # for _, an_instance in self.dense_inst:
            #     features = an_instance.features
            #     all_data.append(features[col_idx])
            all_data = self.dense_data_transpose[col_idx]
            sort_data = sorted(all_data)
            min_rank = int(math.floor((0.5 - 2 * error) * self.count))
            max_rank = int(math.ceil((0.5 + 2 * error) * self.count))
            self.assertTrue(sort_data[min_rank] <= medians[col_name] <= sort_data[max_rank])

    def test_MultivariateStatisticalSummary_not_inst_version(self):

        for col in self.col_index:
            col_name = self.header[col]
            this_data = self.dense_data_transpose[col]
            mean = self.summary_obj_not_inst.get_mean()[col_name]
            var = self.summary_obj_not_inst.get_variance()[col_name]
            max_value = self.summary_obj_not_inst.get_max()[col_name]
            min_value = self.summary_obj_not_inst.get_min()[col_name]

            real_max = np.max(this_data)
            real_min = np.min(this_data)
            real_mean = np.mean(this_data)
            real_var = np.var(this_data)

            self.assertTrue(math.fabs(mean - real_mean) < self.eps)
            self.assertTrue(math.fabs(var - real_var) < self.eps)
            self.assertTrue(max_value == real_max)
            self.assertTrue(min_value == real_min)

    def test_median_not_inst(self):
        error = consts.DEFAULT_RELATIVE_ERROR
        medians = self.summary_obj_not_inst.get_median()

        for col_idx in self.col_index:
            col_name = self.header[col_idx]
            if col_idx not in self.col_index:
                continue
            all_data = self.dense_data_transpose[col_idx]
            sort_data = sorted(all_data)
            min_rank = int(math.floor((0.5 - 2 * error) * self.count))
            max_rank = int(math.ceil((0.5 + 2 * error) * self.count))
            self.assertTrue(sort_data[min_rank] <= medians[col_name] <= sort_data[max_rank])

    def test_quantile_query(self):
        quantile_points = [0.25, 0.5, 0.75, 1.0]
        expect_value = [3, 7, 11, 15]
        for idx, quantile in enumerate(quantile_points):
            quantile_value = self.summary_obj.get_quantile_point(quantile)
            for q_value in quantile_value.values():
                self.assertTrue(q_value == expect_value[idx])

        for idx, quantile in enumerate(quantile_points):
            quantile_value = self.summary_obj_not_inst.get_quantile_point(quantile)
            for q_value in quantile_value.values():
                self.assertTrue(q_value == expect_value[idx])

    def tearDown(self):
        self.dense_table.destroy()
        self.dense_not_inst_table.destroy()


if __name__ == '__main__':
    unittest.main()
