import copy

import numpy as np
import time
import unittest

from sklearn.preprocessing import MinMaxScaler as MMS

from arch.api import session
from federatedml.feature.feature_scale.min_max_scale import MinMaxScale
from federatedml.feature.instance import Instance
from federatedml.param.scale_param import ScaleParam
from federatedml.util.param_extract import ParamExtract


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
        self.table_instance.schema['header'] = ["fid" + str(i) for i in range(len(self.test_data[0]))]

    def print_table(self, table):
        for v in (list(table.collect())):
            print("id:{}, value:{}".format(v[0], v[1].features))

    def data_to_eggroll_table(self, data, jobid, partition=1, work_mode=0):
        session.init(jobid, mode=work_mode)
        data_table = session.parallelize(data, include_key=False)
        return data_table

    def sklearn_attribute_format(self, scaler, feature_range):
        format_output = []
        for i in range(scaler.data_min_.shape[0]):
            col_transform_value = (scaler.data_min_[i], scaler.data_max_[i])
            format_output.append(col_transform_value)

        return format_output

    def get_table_instance_feature(self, table_instance):
        res_list = []
        for k, v in list(table_instance.collect()):
            res_list.append(list(v.features))

        return res_list

    def get_scale_param(self):
        component_param = {
            "method": "standard_scale",
            "mode": "normal",
            "area": "all",
            "scale_column_idx": []
        }
        scale_param = ScaleParam()
        param_extracter = ParamExtract()
        param_extracter.parse_param_from_config(scale_param, component_param)
        return scale_param

    # test with (mode='normal', area='all', feat_upper=None, feat_lower=None)
    def test_fit_instance_default(self):
        scale_param = self.get_scale_param()
        scale_obj = MinMaxScale(scale_param)
        fit_instance = scale_obj.fit(self.table_instance)
        column_min_value = scale_obj.column_min_value
        column_max_value = scale_obj.column_max_value

        scaler = MMS()
        scaler.fit(self.test_data)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             np.around(scaler.transform(self.test_data), 6).tolist())
        data_min = list(scaler.data_min_)
        data_max = list(scaler.data_max_)
        self.assertListEqual(column_min_value, data_min)
        self.assertListEqual(column_max_value, data_max)

        transform_data = scale_obj.transform(self.table_instance)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             self.get_table_instance_feature(transform_data))

    # test with (area="all", upper=2, lower=1):
    def test_fit1(self):
        scale_param = self.get_scale_param()
        scale_param.scale_column_idx = []
        scale_param.feat_upper = 2
        scale_param.feat_lower = 1

        scale_obj = MinMaxScale(scale_param)
        fit_instance = scale_obj.fit(self.table_instance)
        column_min_value = scale_obj.column_min_value
        column_max_value = scale_obj.column_max_value

        for i, line in enumerate(self.test_data):
            for j, value in enumerate(line):
                if value > 2:
                    self.test_data[i][j] = 2
                elif value < 1:
                    self.test_data[i][j] = 1

        scaler = MMS()
        scaler.fit(self.test_data)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             np.around(scaler.transform(self.test_data), 6).tolist())

        data_min = list(scaler.data_min_)
        data_max = list(scaler.data_max_)
        self.assertListEqual(column_min_value, data_min)
        self.assertListEqual(column_max_value, data_max)

        transform_data = scale_obj.transform(self.table_instance)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             self.get_table_instance_feature(transform_data))

    # test with (area="all", upper=[2,2,2,2,2,2], lower=[1,1,1,1,1,1]):
    def test_fit2(self):
        scale_param = self.get_scale_param()
        scale_param.scale_column_idx = []
        scale_param.feat_upper = [2, 2, 2, 2, 2, 2]
        scale_param.feat_lower = [1, 1, 1, 1, 1, 1]

        scale_obj = MinMaxScale(scale_param)
        fit_instance = scale_obj.fit(self.table_instance)
        column_min_value = scale_obj.column_min_value
        column_max_value = scale_obj.column_max_value

        for i, line in enumerate(self.test_data):
            for j, value in enumerate(line):
                if value > 2:
                    self.test_data[i][j] = 2
                elif value < 1:
                    self.test_data[i][j] = 1

        scaler = MMS()
        scaler.fit(self.test_data)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             np.around(scaler.transform(self.test_data), 6).tolist())

        data_min = list(scaler.data_min_)
        data_max = list(scaler.data_max_)
        self.assertListEqual(column_min_value, data_min)
        self.assertListEqual(column_max_value, data_max)

        transform_data = scale_obj.transform(self.table_instance)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             self.get_table_instance_feature(transform_data))

    # test with (area="col", scale_column_idx=[1,2,4], upper=[2,2,2,2,2,2], lower=[1,1,1,1,1,1]):
    def test_fit3(self):
        scale_column_idx = [1, 2, 4]
        scale_param = self.get_scale_param()
        scale_param.area = "col"
        scale_param.feat_upper = [2, 2, 2, 2, 2, 2]
        scale_param.feat_lower = [1, 1, 1, 1, 1, 1]
        scale_param.scale_column_idx = scale_column_idx

        scale_obj = MinMaxScale(scale_param)
        fit_instance = scale_obj.fit(self.table_instance)
        column_min_value = scale_obj.column_min_value
        column_max_value = scale_obj.column_max_value

        raw_data = copy.deepcopy(self.test_data)
        for i, line in enumerate(self.test_data):
            for j, value in enumerate(line):
                if j in scale_column_idx:
                    if value > 2:
                        self.test_data[i][j] = 2
                    elif value < 1:
                        self.test_data[i][j] = 1

        scaler = MMS()
        scaler.fit(self.test_data)
        sklearn_transform_data = np.around(scaler.transform(self.test_data), 6).tolist()
        for i, line in enumerate(sklearn_transform_data):
            for j, cols in enumerate(line):
                if j not in scale_column_idx:
                    sklearn_transform_data[i][j] = raw_data[i][j]

        self.assertListEqual(self.get_table_instance_feature(fit_instance), sklearn_transform_data)

        for i, line in enumerate(sklearn_transform_data):
            for j, cols in enumerate(line):
                if j not in scale_column_idx:
                    sklearn_transform_data[i][j] = raw_data[i][j]

        data_min = list(scaler.data_min_)
        data_max = list(scaler.data_max_)
        self.assertListEqual(column_min_value, data_min)
        self.assertListEqual(column_max_value, data_max)

        transform_data = scale_obj.transform(self.table_instance)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             self.get_table_instance_feature(transform_data))

    # test with (area="col", scale_column_idx=[1,2,4], upper=[2,2,2,2,2,2], lower=[1,1,1,1,1,1]):
    def test_fit4(self):
        scale_column_idx = [1, 2, 4]
        scale_param = self.get_scale_param()
        scale_param.area = "col"
        scale_param.feat_upper = 2
        scale_param.feat_lower = 1
        scale_param.scale_column_idx = scale_column_idx

        scale_obj = MinMaxScale(scale_param)
        fit_instance = scale_obj.fit(self.table_instance)
        column_min_value = scale_obj.column_min_value
        column_max_value = scale_obj.column_max_value

        raw_data = copy.deepcopy(self.test_data)
        for i, line in enumerate(self.test_data):
            for j, value in enumerate(line):
                if j in scale_column_idx:
                    if value > 2:
                        self.test_data[i][j] = 2
                    elif value < 1:
                        self.test_data[i][j] = 1

        scaler = MMS()
        scaler.fit(self.test_data)
        sklearn_transform_data = np.around(scaler.transform(self.test_data), 6).tolist()
        for i, line in enumerate(sklearn_transform_data):
            for j, cols in enumerate(line):
                if j not in scale_column_idx:
                    sklearn_transform_data[i][j] = raw_data[i][j]

        self.assertListEqual(self.get_table_instance_feature(fit_instance), sklearn_transform_data)

        for i, line in enumerate(sklearn_transform_data):
            for j, cols in enumerate(line):
                if j not in scale_column_idx:
                    sklearn_transform_data[i][j] = raw_data[i][j]

        data_min = list(scaler.data_min_)
        data_max = list(scaler.data_max_)
        self.assertListEqual(column_min_value, data_min)
        self.assertListEqual(column_max_value, data_max)

        transform_data = scale_obj.transform(self.table_instance)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             self.get_table_instance_feature(transform_data))

    # test with (area="col", scale_column_idx=[1,2,4], upper=[2,2,2,2,2,2], lower=[1,1,1,1,1,1]):
    def test_fit5(self):
        scale_column_idx = [1, 2, 4]
        scale_param = self.get_scale_param()
        scale_param.mode = "cap"
        scale_param.area = "col"
        scale_param.feat_upper = 0.8
        scale_param.feat_lower = 0.2
        scale_param.scale_column_idx = scale_column_idx

        scale_obj = MinMaxScale(scale_param)
        fit_instance = scale_obj.fit(self.table_instance)
        column_min_value = scale_obj.column_min_value
        column_max_value = scale_obj.column_max_value

        raw_data = copy.deepcopy(self.test_data)
        gt_cap_lower_list = [0, 2, 2, 2, 3, 1]
        gt_cap_upper_list = [1, 8, 8, 8, 7, 8]

        for i, line in enumerate(self.test_data):
            for j, value in enumerate(line):
                if value > gt_cap_upper_list[j]:
                    self.test_data[i][j] = gt_cap_upper_list[j]
                elif value < gt_cap_lower_list[j]:
                    self.test_data[i][j] = gt_cap_lower_list[j]

        scaler = MMS()
        scaler.fit(self.test_data)
        sklearn_transform_data = np.around(scaler.transform(self.test_data), 6).tolist()
        for i, line in enumerate(sklearn_transform_data):
            for j, cols in enumerate(line):
                if j not in scale_column_idx:
                    sklearn_transform_data[i][j] = raw_data[i][j]

        self.assertListEqual(self.get_table_instance_feature(fit_instance), sklearn_transform_data)

        for i, line in enumerate(sklearn_transform_data):
            for j, cols in enumerate(line):
                if j not in scale_column_idx:
                    sklearn_transform_data[i][j] = raw_data[i][j]

        data_min = list(scaler.data_min_)
        data_max = list(scaler.data_max_)
        self.assertListEqual(column_min_value, data_min)
        self.assertListEqual(column_max_value, data_max)

        transform_data = scale_obj.transform(self.table_instance)
        self.assertListEqual(self.get_table_instance_feature(fit_instance),
                             self.get_table_instance_feature(transform_data))


if __name__ == "__main__":
    unittest.main()
