# import math
# import unittest
#
# import numpy as np
#
# from federatedml.feature.binning.quantile_summaries import QuantileSummaries
#
#
# class TestQuantileSummaries(unittest.TestCase):
#     def setUp(self):
#         self.percentile_rate = [90]
#         self.data_num = 1000
#
#         self.table = np.random.randn(self.data_num)
#         compress_thres = 1000
#         head_size = 500
#         self.error = 0.001
#         self.quantile_summaries = QuantileSummaries(compress_thres=compress_thres,
#                                                     head_size=head_size,
#                                                     error=self.error)
#
#     def test_correctness(self):
#         for num in self.table:
#             self.quantile_summaries.insert(num)
#
#         x = sorted(self.table)
#
#         for q_num in self.percentile_rate:
#             percent = q_num / 100
#             sk2 = self.quantile_summaries.query(percent)
#             min_rank = math.floor((percent - 2 * self.error) * self.data_num)
#             max_rank = math.ceil((percent + 2 * self.error) * self.data_num)
#             if min_rank < 0:
#                 min_rank = 0
#             if max_rank > len(x) - 1:
#                 max_rank = len(x) - 1
#             self.assertTrue(x[min_rank] <= sk2 <= x[max_rank])
#
#     def test_multi(self):
#         for n in range(5):
#             self.table = np.random.randn(self.data_num)
#             compress_thres = 10000
#             head_size = 5000
#             self.quantile_summaries = QuantileSummaries(compress_thres=compress_thres,
#                                                         head_size=head_size,
#                                                         error=self.error)
#             self.test_correctness()
#
#
#
# if __name__ == '__main__':
#     unittest.main()
