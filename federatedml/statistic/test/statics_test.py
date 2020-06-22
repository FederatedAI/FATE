import math
import unittest
import uuid

import numpy as np
import time

from arch.api import session
from federatedml.util import consts

session.init("123")

from federatedml.feature.instance import Instance
from federatedml.statistic.statics import MultivariateStatisticalSummary

class TestStatistics(unittest.TestCase):
    def setUp(self):
        self.job_id = str(uuid.uuid1())
        session.init(self.job_id)
        self.eps = 1e-5
        self.count = 10000
        self.feature_num = 100

    def _gen_table_data(self):
        headers = ['x' + str(i) for i in range(self.feature_num)]
        dense_inst = []
        dense_not_inst = []

        original_data = 100 * np.random.random((self.count, self.feature_num))

        for i in range(self.count):
            features = original_data[i, :]
            inst = Instance(features=features)
            dense_inst.append((i, inst))
            dense_not_inst.append((i, features))

        dense_table = session.parallelize(dense_inst, include_key=True, partition=16)
        dense_not_inst_table = session.parallelize(dense_not_inst, include_key=True, partition=16)
        dense_table.schema = {'header': headers}
        dense_not_inst_table.schema = {'header': headers}
        return dense_table, dense_not_inst_table, original_data

    def test_MultivariateStatisticalSummary(self):
        dense_table, dense_not_inst_table, original_data = self._gen_table_data()
        # test max, min
        max_array = np.max(original_data, axis=0)
        min_array = np.min(original_data, axis=0)
        mean_array = np.mean(original_data, axis=0)
        var_array = np.var(original_data, axis=0)
        std_var_array = np.std(original_data, axis=0)

        t0 = time.time()
        header = dense_table.schema['header']
        summary_obj = MultivariateStatisticalSummary(dense_table)
        for idx, col_name in enumerate(header):
            self.assertEqual(summary_obj.get_max()[col_name], max_array[idx])
            self.assertEqual(summary_obj.get_min()[col_name], min_array[idx])
            self.assertTrue(self._float_equal(summary_obj.get_mean()[col_name], mean_array[idx]))
            self.assertTrue(self._float_equal(summary_obj.get_variance()[col_name], var_array[idx]))
            self.assertTrue(self._float_equal(summary_obj.get_std_variance()[col_name], std_var_array[idx]))

        print("total time: {}".format(time.time() - t0))

    def _float_equal(self, x, y):
        if math.fabs(x - y) < consts.FLOAT_ZERO:
            return True
        print(f"x: {x}, y: {y}")
        return False

    def test_median(self):
        error = consts.DEFAULT_RELATIVE_ERROR
        dense_table, dense_not_inst_table, original_data = self._gen_table_data()

        median_array = np.median(original_data, axis=0)
        header = dense_table.schema['header']
        summary_obj = MultivariateStatisticalSummary(dense_table, error=error)

        t0 = time.time()

        summary_obj.get_median()
        quantile_summary_obj = list(summary_obj.binning_obj.summary_dict.values())[0]
        print(f"quantile_summary_obj count: {quantile_summary_obj.count}")
        for stat in quantile_summary_obj.sampled:
            print(stat.__dict__)
        for idx, col_name in enumerate(header):
            self.assertEqual(summary_obj.get_median()[col_name], median_array[idx])
        print("total time: {}".format(time.time() - t0))


    # def test_MultivariateStatisticalSummary_not_inst_version(self):
    #
    #     for col in self.col_index:
    #         col_name = self.header[col]
    #         this_data = self.dense_data_transpose[col]
    #         mean = self.summary_obj_not_inst.get_mean()[col_name]
    #         var = self.summary_obj_not_inst.get_variance()[col_name]
    #         max_value = self.summary_obj_not_inst.get_max()[col_name]
    #         min_value = self.summary_obj_not_inst.get_min()[col_name]
    #
    #         real_max = np.max(this_data)
    #         real_min = np.min(this_data)
    #         real_mean = np.mean(this_data)
    #         real_var = np.var(this_data)
    #
    #         self.assertTrue(math.fabs(mean - real_mean) < self.eps)
    #         self.assertTrue(math.fabs(var - real_var) < self.eps)
    #         self.assertTrue(max_value == real_max)
    #         self.assertTrue(min_value == real_min)
    #
    # def test_median_not_inst(self):
    #     error = consts.DEFAULT_RELATIVE_ERROR
    #     medians = self.summary_obj_not_inst.get_median()
    #
    #     for col_idx in self.col_index:
    #         col_name = self.header[col_idx]
    #         if col_idx not in self.col_index:
    #             continue
    #         all_data = self.dense_data_transpose[col_idx]
    #         sort_data = sorted(all_data)
    #         min_rank = int(math.floor((0.5 - 2 * error) * self.count))
    #         max_rank = int(math.ceil((0.5 + 2 * error) * self.count))
    #         self.assertTrue(sort_data[min_rank] <= medians[col_name] <= sort_data[max_rank])
    #
    # def test_quantile_query(self):
    #     quantile_points = [0.25, 0.5, 0.75, 1.0]
    #     expect_value = [3, 7, 11, 15]
    #     for idx, quantile in enumerate(quantile_points):
    #         quantile_value = self.summary_obj.get_quantile_point(quantile)
    #         for q_value in quantile_value.values():
    #             self.assertTrue(q_value == expect_value[idx])
    #
    #     for idx, quantile in enumerate(quantile_points):
    #         quantile_value = self.summary_obj_not_inst.get_quantile_point(quantile)
    #         for q_value in quantile_value.values():
    #             self.assertTrue(q_value == expect_value[idx])

    def tearDown(self):
        session.stop()
        try:
            session.cleanup("*", self.job_id, True)
        except EnvironmentError:
            pass
        try:
            session.cleanup("*", self.job_id, False)
        except EnvironmentError:
            pass


if __name__ == '__main__':
    unittest.main()
