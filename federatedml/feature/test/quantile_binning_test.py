# #
# #  Copyright 2019 The FATE Authors. All Rights Reserved.
# #
# #  Licensed under the Apache License, Version 2.0 (the "License");
# #  you may not use this file except in compliance with the License.
# #  You may obtain a copy of the License at
# #
# #      http://www.apache.org/licenses/LICENSE-2.0
# #
# #  Unless required by applicable law or agreed to in writing, software
# #  distributed under the License is distributed on an "AS IS" BASIS,
# #  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #  See the License for the specific language governing permissions and
# #  limitations under the License.
# #
#
# import math
# import unittest
#
# import numpy as np
#
# from arch.api import session
#
# session.init("123")
#
# from federatedml.feature.binning.quantile_binning import QuantileBinning
# from federatedml.feature.instance import Instance
# from federatedml.param.feature_binning_param import FeatureBinningParam
# from federatedml.feature.sparse_vector import SparseVector
#
#
# compress_thres = 10000
# head_size = 500
# error = 0.001
# bin_num = 30
#
#
# class TestQuantileBinning(unittest.TestCase):
#     def setUp(self):
#         # eggroll.init("123")
#         self.data_num = 10
#         self.feature_num = 5
#         final_result = []
#         numpy_array = []
#         for i in range(self.data_num):
#             tmp = np.random.rand(self.feature_num)
#             inst = Instance(inst_id=i, features=tmp, label=0)
#             tmp_pair = (str(i), inst)
#             final_result.append(tmp_pair)
#             numpy_array.append(tmp)
#         table = session.parallelize(final_result,
#                                     include_key=True,
#                                     partition=10)
#
#         header = ['x' + str(i) for i in range(self.feature_num)]
#         self.col_dict = {}
#         for idx, h in enumerate(header):
#             self.col_dict[h] = idx
#
#         self.table = table
#         self.table.schema = {'header': header}
#         self.numpy_table = np.array(numpy_array)
#         self.cols = [1, 2]
#         self.used_data_set = []
#         # self.cols = -1
#
#     def _bin_obj_generator(self, abnormal_list=[]):
#
#         bin_param = FeatureBinningParam(method='quantile', compress_thres=compress_thres, head_size=head_size,
#                                         error=error,
#                                         bin_indexes=-1,
#                                         bin_num=bin_num)
#         bin_obj = QuantileBinning(bin_param, abnormal_list=abnormal_list)
#         return bin_obj
#
#     def validate_result(self, table, bin_obj, numpy_table, is_sparse=False, all_data_shape=0):
#         header = table.schema['header']
#         data_num = table.count()
#
#         percent_list = [x / 100 for x in range(100)]
#         total_case = 0
#         error_case = 0
#         for p in percent_list:
#             for bin_idx, col_name in enumerate(header):
#                 x = numpy_table[:, bin_idx]
#                 if is_sparse:
#                     zero_num = all_data_shape - len(x)
#                     if zero_num > 0:
#                         zero_array = np.zeros(zero_num)
#                         x = np.append(x, zero_array)
#                 x = sorted(x)
#                 result = bin_obj.query_quantile_point(table, [col_name], p)
#                 min_rank = int(math.floor(p * data_num - data_num * 2 * error))
#                 max_rank = int(math.ceil(p * data_num + data_num * 2 * error))
#                 if min_rank < 0:
#                     min_rank = 0
#                 if max_rank > len(x) - 1:
#                     max_rank = len(x) - 1
#                 try:
#                     # self.assertTrue(x[min_rank] <= split_points[col_name][bin_idx] <= x[max_rank])
#                     self.assertTrue(x[min_rank] <= result[col_name] <= x[max_rank])
#
#                 except AssertionError:
#                     print("Error in bin: {}, percent: {}".format(col_name, p))
#                     print("result: {}, min_rank: {}, max_rank: {}, x[min_rank]: {}, x[max_rank]: {}".format(
#                         result, min_rank, max_rank, x[min_rank], x[max_rank]
#                     ))
#                     value = list(result.values())[0]
#                     print("Real rank is : {}".format(x.index(value)))
#                     error_case += 1
#                 total_case += 1
#         print("Error rate: {}".format(error_case / total_case))
#
#     def test_one_partition_prob(self):
#         final_result = []
#         numpy_array = []
#         for i in range(self.data_num):
#             tmp = 100 * np.random.rand(self.feature_num)
#             inst = Instance(inst_id=i, features=tmp, label=0)
#             tmp_pair = (str(i), inst)
#             final_result.append(tmp_pair)
#             numpy_array.append(tmp)
#         table = session.parallelize(final_result,
#                                     include_key=True,
#                                     partition=1)
#         header = ['x' + str(i) for i in range(self.feature_num)]
#         numpy_table = np.array(numpy_array)
#         table.schema = {'header': header}
#
#         self.used_data_set.append(table)
#
#         bin_obj = self._bin_obj_generator()
#         bin_obj.fit_split_points(table)
#         self.validate_result(table, bin_obj, numpy_table)
#
#     # def test_sparse_data_prob(self):
#     #     final_result = []
#     #     numpy_array = []
#     #     sparse_inst_shape = self.feature_num + 15
#     #     indices = [x for x in range(self.feature_num + 10)]
#     #     for i in range(self.data_num):
#     #         tmp = 100 * np.random.rand(self.feature_num)
#     #         data_index = np.random.choice(indices, self.feature_num, replace=False)
#     #         data_index = sorted(data_index)
#     #         sparse_inst = SparseVector(data_index, tmp, shape=sparse_inst_shape)
#     #         inst = Instance(inst_id=i, features=sparse_inst, label=0)
#     #         tmp_pair = (str(i), inst)
#     #         final_result.append(tmp_pair)
#     #         n = 0
#     #         pointer = 0
#     #         tmp_array = []
#     #         while n < sparse_inst_shape:
#     #             if n in data_index:
#     #                 tmp_array.append(tmp[pointer])
#     #                 pointer += 1
#     #             else:
#     #                 tmp_array.append(0)
#     #             n += 1
#     #
#     #         numpy_array.append(tmp_array)
#     #     table = eggroll.parallelize(final_result,
#     #                                 include_key=True,
#     #                                 partition=1)
#     #     header = ['x' + str(i) for i in range(sparse_inst_shape)]
#     #     numpy_table = np.array(numpy_array)
#     #     table.schema = {'header': header}
#     #
#     #     self.used_data_set.append(table)
#     #
#     #     bin_obj = self._bin_obj_generator()
#     #     bin_obj.fit_split_points(table)
#     #     self.validate_result(table, bin_obj, numpy_table, is_sparse=True, all_data_shape=sparse_inst_shape)
#     #
#     #     test_array = numpy_table[:, (sparse_inst_shape - 2)]
#     #     null_array = np.array(test_array)
#     #     self.assertTrue(all(null_array == 0))
#
#     def test_sparse_abnormal_data(self):
#         final_result = []
#         numpy_array = []
#         sparse_inst_shape = self.feature_num + 15
#         indices = [x for x in range(self.feature_num + 10)]
#         for i in range(self.data_num):
#             tmp = 100 * np.random.rand(self.feature_num)
#             tmp = [ik for ik in range(self.feature_num)]
#             tmp[i % self.feature_num] = 'nan'
#             # data_index = np.random.choice(indices, self.feature_num, replace=False)
#             # data_index = sorted(data_index)
#             data_index = [idx for idx in range(self.feature_num)]
#             sparse_inst = SparseVector(data_index, tmp, shape=sparse_inst_shape)
#             if i == 0:
#                 aa = sparse_inst.get_data(0, 'a')
#                 print('in for loop: {}, type: {}'.format(aa, type(aa)))
#             inst = Instance(inst_id=i, features=sparse_inst, label=0)
#             tmp_pair = (str(i), inst)
#             final_result.append(tmp_pair)
#             n = 0
#             pointer = 0
#             tmp_array = []
#             while n < sparse_inst_shape:
#                 if n in data_index:
#                     tmp_array.append(tmp[pointer])
#                     pointer += 1
#                 else:
#                     tmp_array.append(0)
#                 n += 1
#             numpy_array.append(tmp_array)
#
#         abnormal_value = final_result[0][1].features.get_data(0, 'a')
#         print('abnormal_value: {}, type: {}'.format(abnormal_value, type(abnormal_value)))
#         table = session.parallelize(final_result,
#                                     include_key=True,
#                                     partition=1)
#         header = ['x' + str(i) for i in range(sparse_inst_shape)]
#         numpy_table = np.array(numpy_array)
#         table.schema = {'header': header}
#         self.used_data_set.append(table)
#
#         bin_obj = self._bin_obj_generator(abnormal_list=['nan'])
#         split_points = bin_obj.fit_split_points(table)
#         print('split_points: {}'.format(split_points))
#         print(numpy_table)
#
#         trans_result = bin_obj.transform(table, transform_cols_idx=-1, transform_type='bin_num')
#         trans_result = trans_result.collect()
#         print('transform result: ')
#         for k, v in trans_result:
#             value = v.features.get_all_data()
#             value_list = []
#             for value_k, value_v in value:
#                 value_list.append((value_k, value_v))
#             print(k, value_list)
#         # self.validate_result(table, bin_obj, numpy_table, is_sparse=True, all_data_shape=sparse_inst_shape)
#
#     def tearDown(self):
#         self.table.destroy()
#
#         for d_table in self.used_data_set:
#             d_table.destroy()
#
#
#
# if __name__ == '__main__':
#     unittest.main()
