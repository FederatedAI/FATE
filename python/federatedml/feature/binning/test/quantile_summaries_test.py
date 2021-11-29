import math
import unittest

import numpy as np

from federatedml.feature.binning.quantile_summaries import QuantileSummaries


class TestQuantileSummaries(unittest.TestCase):
    def setUp(self):
        self.percentile_rate = list(range(0, 100, 1))
        self.data_num = 10000
        np.random.seed(15)
        self.table = np.random.randn(self.data_num)
        compress_thres = 1000
        head_size = 500
        self.error = 0.00001
        self.quantile_summaries = QuantileSummaries(compress_thres=compress_thres,
                                                    head_size=head_size,
                                                    error=self.error)

    def test_correctness(self):
        for num in self.table:
            self.quantile_summaries.insert(num)

        x = sorted(self.table)

        for q_num in self.percentile_rate:
            percent = q_num / 100
            sk2 = self.quantile_summaries.query(percent)
            min_rank = math.floor((percent - 2 * self.error) * self.data_num)
            max_rank = math.ceil((percent + 2 * self.error) * self.data_num)
            if min_rank < 0:
                min_rank = 0
            if max_rank > len(x) - 1:
                max_rank = len(x) - 1
            min_value, max_value = x[min_rank], x[max_rank]
            try:
                self.assertTrue(min_value <= sk2 <= max_value)
            except AssertionError as e:
                print(f"min_value: {min_value}, max_value: {max_value}, sk2: {sk2}, percent: {percent}，"
                      f"total_max_value: {x[-1]}")
                raise AssertionError(e)

    def test_multi(self):
        for n in range(5):
            self.table = np.random.randn(self.data_num)
            compress_thres = 10000
            head_size = 5000
            self.quantile_summaries = QuantileSummaries(compress_thres=compress_thres,
                                                        head_size=head_size,
                                                        error=self.error)
            self.test_correctness()


if __name__ == '__main__':
    unittest.main()
