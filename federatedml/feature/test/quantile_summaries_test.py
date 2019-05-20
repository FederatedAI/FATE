import math
import unittest

import numpy as np

from federatedml.feature.quantile_summaries import QuantileSummaries


class TestQuantileSummaries(unittest.TestCase):
    def setUp(self):
        # self.a = [1, 2, 3, 133, 221, 11, 2, 3, 3, 1, 2, 3, 1, 23, 32] * 100000
        # self.quantile_summary = QuantileSummaries()
        self.percentile_rate = [90]
        self.data_num = 100000
        # self.feature_num = 200
        # result = []
        # for i in range(self.data_num):
        #     result.append(np.random.randn(self.feature_num))
        self.table = np.random.randn(self.data_num)
        compress_thres = 10000
        head_size = 5000
        self.error = 0.001
        self.quantile_summaries = QuantileSummaries(compress_thres=compress_thres,
                                                    head_size=head_size,
                                                    error=self.error)

    def test_correctness(self):
        for num in self.table:
            self.quantile_summaries.insert(num)

        x = sorted(self.table)
        # for idx in range(1, len(x)):
        #     self.assertTrue(x[idx] >= x[idx - 1])

        for q_num in self.percentile_rate:
            percent = q_num / 100
            sk2 = self.quantile_summaries.query(percent)
            min_rank = math.floor((percent - 2 * self.error) * self.data_num)
            max_rank = math.ceil((percent + 2 * self.error) * self.data_num)
            if min_rank < 0:
                min_rank = 0
            if max_rank > len(x) - 1:
                max_rank = len(x) - 1
            found_index = x.index(sk2)
            print("min_rank: {}, found_rank: {}, max_rank: {}".format(
                min_rank, found_index, max_rank
            ))
            self.assertTrue(x[min_rank] <= sk2 <= x[max_rank])

    def test_multi(self):
        for n in range(5):
            self.table = np.random.randn(self.data_num)
            compress_thres = 10000
            head_size = 5000
            self.quantile_summaries = QuantileSummaries(compress_thres=compress_thres,
                                                        head_size=head_size,
                                                        error=self.error)
            self.test_correctness()

            # def test_time_consume(self):
            #     p_r1 = 70
            #     p_r2 = 0.7
            #
            #     t0 = time.time()
            #
            #     sk = np.percentile(self.a, p_r1, interpolation="midpoint")
            #     t1 = time.time()
            #
            #     for num in self.a:
            #         self.quantile_summary.insert(num)
            #     t2 = time.time()
            #     sk2 = self.quantile_summary.query(p_r2)
            #     t3 = time.time()
            #     print('numpy time: {}, insert time: {} summary_time: {}'.format(
            #         t1 - t0, t2 - t1, t3 - t2
            #     ))
            #     print('numpy result: {}, summary result: {}'.format(
            #         sk, sk2
            #     ))


if __name__ == '__main__':
    unittest.main()
