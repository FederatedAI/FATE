import numpy as np
import time
import unittest

from arch.api import eggroll
from federatedml.feature.instance import Instance
from federatedml.feature.standard_scaler import StandardScaler

from sklearn.preprocessing import StandardScaler as SSL
class TestStandardScaler(unittest.TestCase):
    def setUp(self):
        self.test_data = [
                    [0,1.0,10,2,3,1],
                    [1.0,2,9,2,4,2],
                    [0,3.0,8,3,3,3],
                    [1.0,4,7,4,4,4],
                    [1.0,5,6,5,5,5],
                    [1.0,6,5,6,6,-100],
                    [0,7.0,4,7,7,7],
                    [0,8,3.0,8,6,8],
                    [0,9,2,9.0,9,9],
                    [0,10,1,10.0,10,10]
                    ]
        str_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

        
        self.test_instance = []
        for td in self.test_data:
            self.test_instance.append(Instance(features=np.array(td)))
        self.table_instance =  self.data_to_eggroll_table(self.test_instance,  str_time)
    
    def print_table(self, table):
        for v in (list(table.collect())):
            print(v[1].features)

    def data_to_eggroll_table(self, data, jobid, partition=1, work_mode=0):
        eggroll.init(jobid, mode=work_mode)
        data_table = eggroll.parallelize(data, include_key=False, partition = 10)
        return data_table

    def get_table_instance_feature(self, table_instance):
        res_list = []
        for k, v in list(table_instance.collect()):
            res_list.append(list(np.around(v.features, 4)))
        
        return res_list
    
    # test with (with_mean=True, with_std=True):
    def test_fit1(self):
        standard_scaler = StandardScaler(with_mean=True, with_std=True)
        fit_instance, mean, std = standard_scaler.fit(self.table_instance)
        
        scaler = SSL()
        scaler.fit(self.test_data)
        self.assertListEqual(self.get_table_instance_feature(fit_instance), np.around(scaler.transform(self.test_data),4).tolist())
        self.assertListEqual(list(np.around(mean,4)), list(np.around(scaler.mean_,4)))
        self.assertListEqual(list(np.around(std,4)), list(np.around(scaler.scale_,4)))


    # test with (with_mean=False, with_std=True):
    def test_fit2(self):
        standard_scaler = StandardScaler(with_mean=False, with_std=True)
        fit_instance, mean, std = standard_scaler.fit(self.table_instance)
        
        scaler = SSL(with_mean=False)
        scaler.fit(self.test_data)
        self.assertListEqual(self.get_table_instance_feature(fit_instance), np.around(scaler.transform(self.test_data),4).tolist())
        self.assertListEqual(list(np.around(mean,4)), [ 0 for _ in mean])
        self.assertListEqual(list(np.around(std,4)), list(np.around(scaler.scale_,4)))


    # test with (with_mean=True, with_std=False):
    def test_fit3(self):
        standard_scaler = StandardScaler(with_mean=True, with_std=False)
        fit_instance, mean, std = standard_scaler.fit(self.table_instance)
        
        scaler = SSL(with_std=False)
        scaler.fit(self.test_data)
        self.assertListEqual(self.get_table_instance_feature(fit_instance), np.around(scaler.transform(self.test_data),4).tolist())
        self.assertListEqual(list(np.around(mean,4)), list(np.around(scaler.mean_,4)))
        self.assertListEqual(list(np.around(std,4)), [ 1 for _ in std])


    # test with (with_mean=False, with_std=False):
    def test_fit4(self):
        standard_scaler = StandardScaler(with_mean=False, with_std=False)
        fit_instance, mean, std = standard_scaler.fit(self.table_instance)
        
        scaler = SSL(with_mean=False, with_std=False)
        scaler.fit(self.test_data)
        self.assertListEqual(self.get_table_instance_feature(fit_instance), np.around(scaler.transform(self.test_data),4).tolist())
        self.assertEqual(mean, [ 0 for _ in  range(len(self.test_data[0])) ])
        self.assertEqual(std, [ 1 for _ in  range(len(self.test_data[0])) ])


    # test with (with_mean=True, with_std=True):
    def test_transform1(self):
        standard_scaler = StandardScaler(with_mean=True, with_std=True)
        fit_instance, mean, std = standard_scaler.fit(self.table_instance)
        transform_data = standard_scaler.transform(self.table_instance, mean, std)
        self.assertListEqual(self.get_table_instance_feature(transform_data), self.get_table_instance_feature(fit_instance))


    # test with (with_mean=True, with_std=False):
    def test_transform2(self):
        standard_scaler = StandardScaler(with_mean=True, with_std=False)
        fit_instance, mean, std = standard_scaler.fit(self.table_instance)
        transform_data = standard_scaler.transform(self.table_instance, mean, std)
        self.assertListEqual(self.get_table_instance_feature(transform_data), self.get_table_instance_feature(fit_instance))

    # test with (with_mean=False, with_std=True):
    def test_transform3(self):
        standard_scaler = StandardScaler(with_mean=False, with_std=True)
        fit_instance, mean, std = standard_scaler.fit(self.table_instance)
        transform_data = standard_scaler.transform(self.table_instance, mean, std)
        self.assertListEqual(self.get_table_instance_feature(transform_data), self.get_table_instance_feature(fit_instance))

    # test with (with_mean=False, with_std=False):
    def test_transform4(self):
        standard_scaler = StandardScaler(with_mean=False, with_std=False)
        fit_instance, mean, std = standard_scaler.fit(self.table_instance)
        transform_data = standard_scaler.transform(self.table_instance, mean, std)
        self.assertListEqual(self.get_table_instance_feature(transform_data), self.get_table_instance_feature(fit_instance))


if __name__ == "__main__":
    unittest.main()
