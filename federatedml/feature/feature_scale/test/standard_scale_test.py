import copy

import numpy as np
import time
import unittest

from sklearn.preprocessing import StandardScaler as SSL

from arch.api import session
from federatedml.feature.feature_scale.standard_scale import StandardScale
from federatedml.feature.instance import Instance
from federatedml.param.scale_param import ScaleParam
from federatedml.util.param_extract import ParamExtract


class TestStandardScaler(unittest.TestCase):
    def setUp(self):
        self.test_data = [
            [0, 1.0, 10, 2, 3, 1],
            [1.0, 2, 9, 2, 4, 2],
            [0, 3.0, 8, 3, 3, 3],
            [1.0, 4, 7, 4, 4, 4],
            [1.0, 5, 6, 5, 5, 5],
            [1.0, 6, 5, 6, 6, -100],
            [0, 7.0, 4, 7, 7, 7],
            [0, 8, 3.0, 8, 6, 8],
            [0, 9, 2, 9.0, 9, 9],
            [0, 10, 1, 10.0, 10, 10]
        ]
        str_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

        self.test_instance = []
        for td in self.test_data:
            self.test_instance.append(Instance(features=np.array(td)))
        self.table_instance = self.data_to_eggroll_table(self.test_instance, str_time)
        self.table_instance.schema['header'] = ["fid" + str(i) for i in range(len(self.test_data[0]))]

    def print_table(self, table):
        for v in (list(table.collect())):
            print(v[1].features)

    def data_to_eggroll_table(self, data, jobid, partition=1, work_mode=0):
        session.init(jobid, mode=work_mode)
        data_table = session.parallelize(data, include_key=False, partition=10)
        return data_table

    def get_table_instance_feature(self, table_instance):
        res_list = []
        for k, v in list(table_instance.collect()):
            res_list.append(list(np.around(v.features, 4)))

        return res_list

    def get_scale_param(self):
        component_param = {
            "method": "standard_scale",
            "mode": "normal",
            "area": "all",
            "scale_column_idx": [],
            "with_mean": True,
            "with_std": True,
        }
        scale_param = ScaleParam()
        param_extracter = ParamExtract()
        param_extracter.parse_param_from_config(scale_param, component_param)
        return scale_param

    # test with (with_mean=True, with_std=True):
    def test_fit1(self):
        scale_param = self.get_scale_param()
        standard_scaler = StandardScale(scale_param)
        fit_instance = standard_scaler.fit(self.table_instance)
        mean = standard_scaler.mean
        std = standard_scaler.std

        scaler = SSL()
        scaler.fit(self.test_data)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             np.around(scaler.transform(self.test_data), 4).tolist())
        self.assertListEqual(list(np.around(mean, 4)), list(np.around(scaler.mean_, 4)))
        self.assertListEqual(list(np.around(std, 4)), list(np.around(scaler.scale_, 4)))

    # test with (with_mean=False, with_std=True):
    def test_fit2(self):
        scale_param = self.get_scale_param()
        scale_param.with_mean = False

        standard_scaler = StandardScale(scale_param)
        fit_instance = standard_scaler.fit(self.table_instance)
        mean = standard_scaler.mean
        std = standard_scaler.std

        scaler = SSL(with_mean=False)
        scaler.fit(self.test_data)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             np.around(scaler.transform(self.test_data), 4).tolist())
        self.assertListEqual(list(np.around(mean, 4)), [0 for _ in mean])
        self.assertListEqual(list(np.around(std, 4)), list(np.around(scaler.scale_, 4)))

    # test with (with_mean=True, with_std=False):
    def test_fit3(self):
        scale_param = self.get_scale_param()
        scale_param.with_std = False

        standard_scaler = StandardScale(scale_param)
        fit_instance = standard_scaler.fit(self.table_instance)
        mean = standard_scaler.mean
        std = standard_scaler.std

        scaler = SSL(with_std=False)
        scaler.fit(self.test_data)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             np.around(scaler.transform(self.test_data), 4).tolist())
        self.assertListEqual(list(np.around(mean, 4)), list(np.around(scaler.mean_, 4)))
        self.assertListEqual(list(np.around(std, 4)), [1 for _ in std])

    # test with (with_mean=False, with_std=False):
    def test_fit4(self):
        scale_param = self.get_scale_param()
        scale_param.with_std = False
        scale_param.with_mean = False

        standard_scaler = StandardScale(scale_param)
        fit_instance = standard_scaler.fit(self.table_instance)
        mean = standard_scaler.mean
        std = standard_scaler.std

        scaler = SSL(with_mean=False, with_std=False)
        scaler.fit(self.test_data)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             np.around(scaler.transform(self.test_data), 4).tolist())
        self.assertEqual(mean, [0 for _ in range(len(self.test_data[0]))])
        self.assertEqual(std, [1 for _ in range(len(self.test_data[0]))])

    # test with (area="all", scale_column_idx=[], with_mean=True, with_std=True):
    def test_fit5(self):
        scale_param = self.get_scale_param()
        scale_param.scale_column_idx = []

        standard_scaler = StandardScale(scale_param)
        fit_instance = standard_scaler.fit(self.table_instance)
        mean = standard_scaler.mean
        std = standard_scaler.std

        scaler = SSL()
        scaler.fit(self.test_data)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             np.around(scaler.transform(self.test_data), 4).tolist())
        self.assertListEqual(list(np.around(mean, 4)), list(np.around(scaler.mean_, 4)))
        self.assertListEqual(list(np.around(std, 4)), list(np.around(scaler.scale_, 4)))

    # test with (area="col", scale_column_idx=[], with_mean=True, with_std=True):
    def test_fit6(self):
        scale_param = self.get_scale_param()
        scale_param.scale_column_idx = []
        scale_param.area = "col"

        standard_scaler = StandardScale(scale_param)
        fit_instance = standard_scaler.fit(self.table_instance)
        mean = standard_scaler.mean
        std = standard_scaler.std

        scaler = SSL()
        scaler.fit(self.test_data)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             np.around(self.test_data, 4).tolist())
        self.assertListEqual(list(np.around(mean, 4)), list(np.around(scaler.mean_, 4)))
        self.assertListEqual(list(np.around(std, 4)), list(np.around(scaler.scale_, 4)))

    # test with (area="all", upper=2, lower=1, with_mean=False, with_std=False):
    def test_fit7(self):
        scale_param = self.get_scale_param()
        scale_param.scale_column_idx = []
        scale_param.feat_upper = 2
        scale_param.feat_lower = 1
        scale_param.with_mean = False
        scale_param.with_std = False

        standard_scaler = StandardScale(scale_param)
        fit_instance = standard_scaler.fit(self.table_instance)
        mean = standard_scaler.mean
        std = standard_scaler.std
        column_max_value = standard_scaler.column_max_value
        column_min_value = standard_scaler.column_min_value

        for i, line in enumerate(self.test_data):
            for j, value in enumerate(line):
                if value > 2:
                    self.test_data[i][j] = 2
                elif value < 1:
                    self.test_data[i][j] = 1

        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             np.around(self.test_data, 4).tolist())
        self.assertEqual(mean, [0 for _ in range(len(self.test_data[0]))])
        self.assertEqual(std, [1 for _ in range(len(self.test_data[0]))])
        self.assertEqual(column_max_value, [1, 2, 2, 2, 2, 2])
        self.assertEqual(column_min_value, [1, 1, 1, 2, 2, 1])

    # test with (area="all", upper=[2,2,2,2,2,2], lower=[1,1,1,1,1,1], with_mean=False, with_std=False):
    def test_fit8(self):
        scale_param = self.get_scale_param()
        scale_param.scale_column_idx = []
        scale_param.feat_upper = [2, 2, 2, 2, 2, 2]
        scale_param.feat_lower = [1, 1, 1, 1, 1, 1]
        scale_param.with_mean = False
        scale_param.with_std = False

        standard_scaler = StandardScale(scale_param)
        fit_instance = standard_scaler.fit(self.table_instance)
        mean = standard_scaler.mean
        std = standard_scaler.std
        column_max_value = standard_scaler.column_max_value
        column_min_value = standard_scaler.column_min_value

        for i, line in enumerate(self.test_data):
            for j, value in enumerate(line):
                if value > 2:
                    self.test_data[i][j] = 2
                elif value < 1:
                    self.test_data[i][j] = 1

        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             np.around(self.test_data, 4).tolist())
        self.assertEqual(mean, [0 for _ in range(len(self.test_data[0]))])
        self.assertEqual(std, [1 for _ in range(len(self.test_data[0]))])
        self.assertEqual(column_max_value, [1, 2, 2, 2, 2, 2])
        self.assertEqual(column_min_value, [1, 1, 1, 2, 2, 1])

    # test with (area="col", upper=[2,2,2,2,2,2], lower=[1,1,1,1,1,1], scale_column_idx=[1,2,4], with_mean=True, with_std=True):
    def test_fit9(self):
        scale_column_idx = [1, 2, 4]

        scale_param = self.get_scale_param()
        scale_param.scale_column_idx = []
        scale_param.feat_upper = [2, 2, 2, 2, 2, 2]
        scale_param.feat_lower = [1, 1, 1, 1, 1, 1]
        scale_param.with_mean = True
        scale_param.with_std = True
        scale_param.scale_column_idx = scale_column_idx
        scale_param.area = "col"

        standard_scaler = StandardScale(scale_param)
        fit_instance = standard_scaler.fit(self.table_instance)
        mean = standard_scaler.mean
        std = standard_scaler.std
        column_max_value = standard_scaler.column_max_value
        column_min_value = standard_scaler.column_min_value

        raw_data = copy.deepcopy(self.test_data)
        for i, line in enumerate(self.test_data):
            for j, value in enumerate(line):
                if j in scale_column_idx:
                    if value > 2:
                        self.test_data[i][j] = 2
                    elif value < 1:
                        self.test_data[i][j] = 1

        scaler = SSL(with_mean=True, with_std=True)
        scaler.fit(self.test_data)
        transform_data = np.around(scaler.transform(self.test_data), 4).tolist()

        for i, line in enumerate(transform_data):
            for j, cols in enumerate(line):
                if j not in scale_column_idx:
                    transform_data[i][j] = raw_data[i][j]

        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             transform_data)
        self.assertListEqual(list(np.around(mean, 6)), list(np.around(scaler.mean_, 6)))
        self.assertListEqual(list(np.around(std, 6)), list(np.around(scaler.scale_, 6)))
        self.assertEqual(column_max_value, [1, 2, 2, 10, 2, 10])
        self.assertEqual(column_min_value, [0, 1, 1, 2, 2, -100])

        raw_data_transform = standard_scaler.transform(self.table_instance)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             self.get_table_instance_feature(raw_data_transform))

    # test with (mode="cap", area="col", upper=0.8, lower=0.2, scale_column_idx=[1,2,4], with_mean=True, with_std=True):
    def test_fit10(self):
        scale_column_idx = [1, 2, 4]

        scale_param = self.get_scale_param()
        scale_param.scale_column_idx = []
        scale_param.feat_upper = 0.8
        scale_param.feat_lower = 0.2
        scale_param.with_mean = True
        scale_param.with_std = True
        scale_param.mode = "cap"
        scale_param.scale_column_idx = scale_column_idx
        scale_param.area = "col"

        standard_scaler = StandardScale(scale_param)
        fit_instance = standard_scaler.fit(self.table_instance)
        mean = standard_scaler.mean
        std = standard_scaler.std
        column_max_value = standard_scaler.column_max_value
        column_min_value = standard_scaler.column_min_value

        gt_cap_lower_list = [0, 2, 2, 2, 3, 1]
        gt_cap_upper_list = [1, 8, 8, 8, 7, 8]
        raw_data = copy.deepcopy(self.test_data)
        for i, line in enumerate(self.test_data):
            for j, value in enumerate(line):
                if j in scale_column_idx:
                    if value > gt_cap_upper_list[j]:
                        self.test_data[i][j] = gt_cap_upper_list[j]
                    elif value < gt_cap_lower_list[j]:
                        self.test_data[i][j] = gt_cap_lower_list[j]

        scaler = SSL(with_mean=True, with_std=True)
        scaler.fit(self.test_data)
        transform_data = np.around(scaler.transform(self.test_data), 4).tolist()

        for i, line in enumerate(transform_data):
            for j, cols in enumerate(line):
                if j not in scale_column_idx:
                    transform_data[i][j] = raw_data[i][j]

        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             transform_data)
        self.assertEqual(column_max_value, gt_cap_upper_list)
        self.assertEqual(column_min_value, gt_cap_lower_list)

        self.assertListEqual(list(np.around(mean, 6)), list(np.around(scaler.mean_, 6)))
        self.assertListEqual(list(np.around(std, 6)), list(np.around(scaler.scale_, 6)))

        raw_data_transform = standard_scaler.transform(self.table_instance)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             self.get_table_instance_feature(raw_data_transform))

    # test with (with_mean=True, with_std=True):
    def test_transform1(self):
        scale_param = self.get_scale_param()

        standard_scaler = StandardScale(scale_param)
        fit_instance = standard_scaler.fit(self.table_instance)
        transform_data = standard_scaler.transform(self.table_instance)
        self.assertListEqual(self.get_table_instance_feature(transform_data),
                             self.get_table_instance_feature(fit_instance))

    # test with (with_mean=True, with_std=False):
    def test_transform2(self):
        scale_param = self.get_scale_param()
        scale_param.with_std = False

        standard_scaler = StandardScale(scale_param)
        fit_instance = standard_scaler.fit(self.table_instance)
        transform_data = standard_scaler.transform(self.table_instance)
        self.assertListEqual(self.get_table_instance_feature(transform_data),
                             self.get_table_instance_feature(fit_instance))

    # test with (with_mean=False, with_std=True):
    def test_transform3(self):
        scale_param = self.get_scale_param()
        scale_param.with_mean = False

        standard_scaler = StandardScale(scale_param)
        fit_instance = standard_scaler.fit(self.table_instance)
        transform_data = standard_scaler.transform(self.table_instance)
        self.assertListEqual(self.get_table_instance_feature(transform_data),
                             self.get_table_instance_feature(fit_instance))

    # test with (with_mean=False, with_std=False):
    def test_transform4(self):
        scale_param = self.get_scale_param()
        scale_param.with_mean = False
        scale_param.with_std = False

        standard_scaler = StandardScale(scale_param)
        fit_instance = standard_scaler.fit(self.table_instance)
        transform_data = standard_scaler.transform(self.table_instance)
        self.assertListEqual(self.get_table_instance_feature(transform_data),
                             self.get_table_instance_feature(fit_instance))

    # test with (area='all', scale_column_idx=[], with_mean=False, with_std=False):
    def test_transform5(self):
        scale_param = self.get_scale_param()
        scale_param.with_mean = False
        scale_param.with_std = False
        scale_param.area = "all"
        scale_param.scale_column_idx = []

        standard_scaler = StandardScale(scale_param)
        fit_instance = standard_scaler.fit(self.table_instance)
        transform_data = standard_scaler.transform(self.table_instance)
        self.assertListEqual(self.get_table_instance_feature(transform_data),
                             self.get_table_instance_feature(fit_instance))

    # test with (area='col', with_mean=[], with_std=False):
    def test_transform6(self):
        scale_param = self.get_scale_param()
        scale_param.with_mean = False
        scale_param.with_std = False
        scale_param.area = "col"
        scale_param.scale_column_idx = []

        standard_scaler = StandardScale(scale_param)
        fit_instance = standard_scaler.fit(self.table_instance)
        transform_data = standard_scaler.transform(self.table_instance)
        self.assertListEqual(self.get_table_instance_feature(transform_data),
                             self.get_table_instance_feature(fit_instance))

    def test_cols_select_fit_and_transform(self):
        scale_param = self.get_scale_param()
        scale_param.scale_column_idx = [1, 2, 4]
        scale_param.area = "col"
        standard_scaler = StandardScale(scale_param)
        fit_data = standard_scaler.fit(self.table_instance)

        scale_column_idx = standard_scaler.scale_column_idx

        scaler = SSL(with_mean=True, with_std=True)
        scaler.fit(self.test_data)
        transform_data = np.around(scaler.transform(self.test_data), 4).tolist()

        for i, line in enumerate(transform_data):
            for j, cols in enumerate(line):
                if j not in scale_column_idx:
                    transform_data[i][j] = self.test_data[i][j]

        self.assertListEqual(self.get_table_instance_feature(fit_data),
                             transform_data)

        std_scale_transform_data = standard_scaler.transform(self.table_instance)
        self.assertListEqual(self.get_table_instance_feature(std_scale_transform_data),
                             transform_data)

    def test_cols_select_fit_and_transform_repeat(self):
        scale_param = self.get_scale_param()
        scale_param.scale_column_idx = [1, 1, 2, 2, 4, 5, 5]
        scale_param.area = "col"
        standard_scaler = StandardScale(scale_param)
        fit_data = standard_scaler.fit(self.table_instance)
        scale_column_idx = standard_scaler.scale_column_idx

        scaler = SSL(with_mean=True, with_std=True)
        scaler.fit(self.test_data)
        transform_data = np.around(scaler.transform(self.test_data), 4).tolist()

        for i, line in enumerate(transform_data):
            for j, cols in enumerate(line):
                if j not in scale_column_idx:
                    transform_data[i][j] = self.test_data[i][j]

        self.assertListEqual(self.get_table_instance_feature(fit_data),
                             transform_data)

        std_scale_transform_data = standard_scaler.transform(self.table_instance)
        self.assertListEqual(self.get_table_instance_feature(std_scale_transform_data),
                             transform_data)


if __name__ == "__main__":
    unittest.main()
