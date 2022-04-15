import math
import time
import unittest
import uuid

import numpy as np

from fate_arch.session import computing_session as session
from federatedml.util import consts

session.init("123")

from federatedml.feature.instance import Instance
from federatedml.statistic.statics import MultivariateStatisticalSummary


class TestStatistics(unittest.TestCase):
    def setUp(self):
        self.job_id = str(uuid.uuid1())
        session.init(self.job_id)
        self.eps = 1e-5
        self.count = 1000
        self.feature_num = 100
        self._dense_table, self._dense_not_inst_table, self._original_data = None, None, None

    def _gen_table_data(self):
        if self._dense_table is not None:
            return self._dense_table, self._dense_not_inst_table, self._original_data
        headers = ['x' + str(i) for i in range(self.feature_num)]
        dense_inst = []
        dense_not_inst = []

        original_data = 100 * np.random.random((self.count, self.feature_num))
        # original_data = 100 * np.zeros((self.count, self.feature_num))

        for i in range(self.count):
            features = original_data[i, :]
            inst = Instance(features=features)
            dense_inst.append((i, inst))
            dense_not_inst.append((i, features))

        dense_table = session.parallelize(dense_inst, include_key=True, partition=16)
        dense_not_inst_table = session.parallelize(dense_not_inst, include_key=True, partition=16)
        dense_table.schema = {'header': headers}
        dense_not_inst_table.schema = {'header': headers}
        self._dense_table, self._dense_not_inst_table, self._original_data = \
            dense_table, dense_not_inst_table, original_data
        return dense_table, dense_not_inst_table, original_data

    def _gen_missing_table(self):
        headers = ['x' + str(i) for i in range(self.feature_num)]
        dense_inst = []
        dense_not_inst = []

        original_data = 100 * np.random.random((self.count, self.feature_num))

        for i in range(self.count):
            features = original_data[i, :]
            if i % 2 == 0:
                features = np.array([np.nan] * self.feature_num)
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
        summary_obj = MultivariateStatisticalSummary(dense_table)
        self._test_min_max(summary_obj, original_data, dense_table)
        self._test_min_max(summary_obj, original_data, dense_not_inst_table)

    def _test_min_max(self, summary_obj, original_data, data_table):
        # test max, min
        max_array = np.max(original_data, axis=0)
        min_array = np.min(original_data, axis=0)
        mean_array = np.mean(original_data, axis=0)
        var_array = np.var(original_data, axis=0)
        std_var_array = np.std(original_data, axis=0)

        t0 = time.time()
        header = data_table.schema['header']
        for idx, col_name in enumerate(header):
            self.assertEqual(summary_obj.get_max()[col_name], max_array[idx])
            self.assertEqual(summary_obj.get_min()[col_name], min_array[idx])
            self.assertTrue(self._float_equal(summary_obj.get_mean()[col_name], mean_array[idx]))
            self.assertTrue(self._float_equal(summary_obj.get_variance()[col_name], var_array[idx]))
            self.assertTrue(self._float_equal(summary_obj.get_std_variance()[col_name], std_var_array[idx]))

        print("max value etc, total time: {}".format(time.time() - t0))

    def _float_equal(self, x, y, error=1e-6):
        if math.fabs(x - y) < error:
            return True
        print(f"x: {x}, y: {y}")
        return False

    # def test_median(self):
    #     error = 0
    #     dense_table, dense_not_inst_table, original_data = self._gen_table_data()
    #
    #     sorted_matrix = np.sort(original_data, axis=0)
    #     median_array = sorted_matrix[self.count // 2, :]
    #     header = dense_table.schema['header']
    #     summary_obj = MultivariateStatisticalSummary(dense_table, error=error)
    #     t0 = time.time()
    #
    #     for idx, col_name in enumerate(header):
    #         self.assertTrue(self._float_equal(summary_obj.get_median()[col_name],
    #                                           median_array[idx]))
    #     print("median interface, total time: {}".format(time.time() - t0))
    #
    #     summary_obj_2 = MultivariateStatisticalSummary(dense_not_inst_table, error=error)
    #     t0 = time.time()
    #     for idx, col_name in enumerate(header):
    #         self.assertTrue(self._float_equal(summary_obj_2.get_median()[col_name],
    #                                           median_array[idx]))
    #     print("median interface, total time: {}".format(time.time() - t0))
    #
    # def test_quantile_query(self):
    #
    #     dense_table, dense_not_inst_table, original_data = self._gen_table_data()
    #
    #     quantile_points = [0.25, 0.5, 0.75, 1.0]
    #     quantile_array = np.quantile(original_data, quantile_points, axis=0)
    #     summary_obj = MultivariateStatisticalSummary(dense_table, error=0)
    #     header = dense_table.schema['header']
    #
    #     t0 = time.time()
    #     for q_idx, q in enumerate(quantile_points):
    #         for idx, col_name in enumerate(header):
    #             self.assertTrue(self._float_equal(summary_obj.get_quantile_point(q)[col_name],
    #                                               quantile_array[q_idx][idx],
    #                                               error=3))
    #     print("quantile interface, total time: {}".format(time.time() - t0))
    #
    # def test_missing_value(self):
    #     dense_table, dense_not_inst_table, original_data = self._gen_missing_table()
    #     summary_obj = MultivariateStatisticalSummary(dense_table, error=0)
    #     t0 = time.time()
    #     missing_result = summary_obj.get_missing_ratio()
    #     for col_name, missing_ratio in missing_result.items():
    #         self.assertEqual(missing_ratio, 0.5, msg="missing ratio should be 0.5")
    #     print("calculate missing ratio, total time: {}".format(time.time() - t0))

    def test_moment(self):
        dense_table, dense_not_inst_table, original_data = self._gen_table_data()
        summary_obj = MultivariateStatisticalSummary(dense_table, error=0, stat_order=4, bias=False)
        header = dense_table.schema['header']
        from scipy import stats
        moment_3 = stats.moment(original_data, 3, axis=0)
        moment_4 = stats.moment(original_data, 4, axis=0)
        skewness = stats.skew(original_data, axis=0, bias=False)
        kurtosis = stats.kurtosis(original_data, axis=0, bias=False)

        summary_moment_3 = summary_obj.get_statics("moment_3")
        summary_moment_4 = summary_obj.get_statics("moment_4")
        static_skewness = summary_obj.get_statics("skewness")
        static_kurtosis = summary_obj.get_statics("kurtosis")

        # print(f"moment: {summary_moment_4}, moment_2: {moment_4}")
        for idx, col_name in enumerate(header):
            self.assertTrue(self._float_equal(summary_moment_3[col_name],
                                              moment_3[idx]))
            self.assertTrue(self._float_equal(summary_moment_4[col_name],
                                              moment_4[idx]))
            self.assertTrue(self._float_equal(static_skewness[col_name],
                                              skewness[idx]))
            self.assertTrue(self._float_equal(static_kurtosis[col_name],
                                              kurtosis[idx]))

    def tearDown(self):
        session.stop()


if __name__ == '__main__':
    unittest.main()
