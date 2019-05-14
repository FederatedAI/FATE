import numpy as np
import time
import unittest

from arch.api import eggroll
from federatedml.feature.instance import Instance
from federatedml.feature.min_max_scaler import MinMaxScaler

from sklearn.preprocessing import MinMaxScaler as MMS


class TestMinMaxScaler(unittest.TestCase):
    def setUp(self):
        self.test_data = [
            [0, 1, 10, 2, 3, 1],
            [1, 2, 9, 2, 4, 2],
            [0, 3, 8, 3, 3, 3],
            [1, 4, 7, 4, 4, 4],
            [1, 5, 6, 5, 5, 5],
            [1, 6, 5, 6, 6, -100],
            [0, 7, 4, 7, 7, 7],
            [0, 8, 3, 8, 6, 8],
            [0, 9, 2, 9, 9, 9],
            [0, 10, 1, 10, 10, 10]
        ]
        str_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

        self.test_instance = []
        for td in self.test_data:
            # self.test_instance.append(Instance(features=td))
            self.test_instance.append(Instance(features=np.array(td, dtype=float)))
        self.table_instance = self.data_to_eggroll_table(self.test_instance, str_time)

    def print_table(self, table):
        for v in (list(table.collect())):
            print(v)

    def data_to_eggroll_table(self, data, jobid, partition=1, work_mode=0):
        eggroll.init(jobid, mode=work_mode)
        data_table = eggroll.parallelize(data, include_key=False)
        return data_table

    def sklearn_attribute_format(self, scaler, feature_range):
        format_output = []
        for i in range(scaler.data_min_.shape[0]):
            col_transform_value = (scaler.data_min_[i], scaler.data_max_[i], feature_range[0], feature_range[1])
            format_output.append(col_transform_value)

        return format_output

    def get_table_instance_feature(self, table_instance):
        res_list = []
        for k, v in list(table_instance.collect()):
            res_list.append(list(v.features))

        return res_list

    # test with (mode='normal', area='all', feat_upper=None, feat_lower=None, out_upper=None, out_lower=None)
    def test_fit_instance_default(self):
        min_max_scaler = MinMaxScaler(mode='normal', area='all', feat_upper=None, feat_lower=None, out_upper=None,
                                      out_lower=None)
        fit_instance, cols_transform_value = min_max_scaler.fit(self.table_instance)

        scaler = MMS()
        scaler.fit(self.test_data)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             np.around(scaler.transform(self.test_data), 4).tolist())
        self.assertListEqual(cols_transform_value, self.sklearn_attribute_format(scaler, [0, 1]))

    # test with (mode='normal', area='all', feat_upper=None, feat_lower=None, out_upper=2, out_lower=-1):
    def test_fit_out(self):
        min_max_scaler = MinMaxScaler(mode='normal', area='all', feat_upper=None, feat_lower=None, out_upper=2,
                                      out_lower=-1)
        fit_data, cols_transform_value = min_max_scaler.fit(self.table_instance)

        feature_range = (-1, 2)
        scaler = MMS(feature_range)
        scaler.fit(self.test_data)
        self.assertListEqual(self.get_table_instance_feature(fit_data),
                             np.around(scaler.transform(self.test_data), 4).tolist())
        self.assertListEqual(cols_transform_value, self.sklearn_attribute_format(scaler, feature_range))

    def test_fit_feat(self):
        feat_upper = 8
        feat_lower = 3
        out_upper = 2
        out_lower = -1
        min_max_scaler = MinMaxScaler(mode='normal', area='all', feat_upper=feat_upper, feat_lower=feat_lower,
                                      out_upper=out_upper, out_lower=out_lower)
        fit_data, cols_transform_value = min_max_scaler.fit(self.table_instance)

        new_data = []
        for data in self.test_data:
            tmp_data = []
            for i in range(len(data)):
                if data[i] > feat_upper:
                    tmp_data.append(feat_upper)
                elif data[i] < feat_lower:
                    tmp_data.append(feat_lower)
                else:
                    tmp_data.append(data[i])

            new_data.append(tmp_data)

        feature_range = (out_lower, out_upper)
        scaler = MMS(feature_range)
        scaler.fit(new_data)
        self.assertListEqual(self.get_table_instance_feature(fit_data),
                             np.around(scaler.transform(new_data), 4).tolist())
        sklearn_res = self.sklearn_attribute_format(scaler, [out_lower, out_upper])
        # self.assertListEqual(cols_transform_value, [(feat_lower, feat_upper, out_lower, out_upper)])
        for i in range(len(sklearn_res)):
            if sklearn_res[i][0] != feat_lower:
                tmp_res = list(sklearn_res[i])
                tmp_res[0] = feat_lower
                sklearn_res[i] = tuple(tmp_res)
            if sklearn_res[i][1] != feat_upper:
                tmp_res = list(sklearn_res[i])
                tmp_res[1] = feat_upper
                sklearn_res[i] = tuple(tmp_res)

        self.assertListEqual(cols_transform_value, sklearn_res)

    def test_fit_col_default(self):
        min_max_scaler = MinMaxScaler(mode='normal', area='col', feat_upper=None, feat_lower=None, out_upper=None,
                                      out_lower=None)
        fit_data, cols_transform_value = min_max_scaler.fit(self.table_instance)

        scaler = MMS()
        scaler.fit(self.test_data)
        self.assertListEqual(self.get_table_instance_feature(fit_data),
                             np.around(scaler.transform(self.test_data), 4).tolist())
        self.assertListEqual(cols_transform_value, self.sklearn_attribute_format(scaler, [0, 1]))

    # test with (mode='normal', area='all', feat_upper=None, feat_lower=None, out_upper=2, out_lower=-1):
    def test_fit_col_out(self):
        min_max_scaler = MinMaxScaler(mode='normal', area='col', feat_upper=None, feat_lower=None, out_upper=2,
                                      out_lower=-1)
        fit_data, cols_transform_value = min_max_scaler.fit(self.table_instance)

        feature_range = (-1, 2)
        scaler = MMS(feature_range)
        scaler.fit(self.test_data)
        self.assertListEqual(self.get_table_instance_feature(fit_data),
                             np.around(scaler.transform(self.test_data), 4).tolist())
        self.assertListEqual(cols_transform_value, self.sklearn_attribute_format(scaler, feature_range))

    def test_fit_col_feat(self):
        feat_upper = [0.8, 5, 6, 7, 8, 9]
        feat_lower = [0.5, 2, 3, 3, 4, 4.0]
        out_upper = 2.0
        out_lower = -1.0
        min_max_scaler = MinMaxScaler(mode='normal', area='col', feat_upper=feat_upper, feat_lower=feat_lower,
                                      out_upper=out_upper, out_lower=out_lower)
        fit_data, cols_transform_value = min_max_scaler.fit(self.table_instance)

        new_data = []
        for data in self.test_data:
            tmp_data = []
            for i in range(len(data)):
                if data[i] > feat_upper[i]:
                    tmp_data.append(feat_upper[i])
                elif data[i] < feat_lower[i]:
                    tmp_data.append(feat_lower[i])
                else:
                    tmp_data.append(data[i])

            new_data.append(tmp_data)

        feature_range = (out_lower, out_upper)
        scaler = MMS(feature_range)
        scaler.fit(new_data)
        self.assertListEqual(self.get_table_instance_feature(fit_data),
                             np.around(scaler.transform(new_data), 4).tolist())

    def test_transform_all(self):
        feat_upper = 8
        feat_lower = 3
        out_upper = 2
        out_lower = -1
        min_max_scaler = MinMaxScaler(mode='normal', area='all', feat_upper=feat_upper, feat_lower=feat_lower,
                                      out_upper=out_upper, out_lower=out_lower)
        fit_data, cols_transform_value = min_max_scaler.fit(self.table_instance)

        transform_data = min_max_scaler.transform(self.table_instance, cols_transform_value)

        self.assertListEqual(self.get_table_instance_feature(fit_data), self.get_table_instance_feature(transform_data))


if __name__ == "__main__":
    unittest.main()
