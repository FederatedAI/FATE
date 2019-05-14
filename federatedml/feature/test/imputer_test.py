import functools
import numpy as np
import sys
import time
import unittest

from arch.api import eggroll
from federatedml.feature.instance import Instance
from federatedml.feature.imputer import Imputer

from sklearn.preprocessing import StandardScaler as SSL
class TestMinMaxScaler(unittest.TestCase):
    def setUp(self):
        local_time = time.localtime(time.time())
        str_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
        
        self.test_data = [
        [  "0.254879",       "na", "0.209656",       "NA","-0.441366",       "Na","-0.485934",       "nA","-0.287570","-0.733474"],
        [ "-1.142928",         "","-1.166747","-0.923578", "0.628230","-1.021418","-1.111867","-0.959523","-0.096672","-0.121683"],
        [ "-1.451067","-1.406518",     "none","-1.092337",     "None","-1.168557","-1.305831","-1.745063","-0.499499","-0.302893"],
        [ "-0.879933",     "null","-0.877527","-0.780484","-1.037534","-0.483880","-0.555498","-0.768581", "0.433960","-0.200928"],
        [  "0.426758", "0.723479", "0.316885", "0.287273", "1.000835", "0.962702", "1.077099", "1.053586", "2.996525", "0.961696"],
        [  "0.963102", "1.467675", "0.829202", "0.772457","-0.038076","-0.468613","-0.307946","-0.015321","-0.641864","-0.247477"],
        [ "-0.662496", "0.212149","-0.620475","-0.632995","-0.327392","-0.385278","-0.077665","-0.730362", "0.217178","-0.061280"],
        [ "-0.453343","-2.147457","-0.473631","-0.483572", "0.558093","-0.740244","-0.896170","-0.617229","-0.308601","-0.666975"],
        [ "-0.606584","-0.971725","-0.678558","-0.591332","-0.963013","-1.302401","-1.212855","-1.321154","-1.591501","-1.230554"],
        [ "-0.583805","-0.193332","-0.633283","-0.560041","-0.349310","-0.519504","-0.610669","-0.929526","-0.196974","-0.151608"]
        ]
        self.test_instance = []
        for td in self.test_data:
            self.test_instance.append(td)
        self.table_instance =  self.data_to_eggroll_table(self.test_instance,  str_time)
    
    def print_table(self, table):
        for v in (list(table.collect())):
            print(v[1].features)

    def data_to_eggroll_table(self, data, jobid, partition=1, work_mode=0):
        eggroll.init(jobid, mode=work_mode)
        data_table = eggroll.parallelize(data, include_key=False, partition = 10)
        return data_table

    def table_to_list(self, table_instance):
        res_list = []
        for k, v in list(table_instance.collect()):
            res_list.append(list(v))
        
        return res_list
    
    def fit_test_data(self, data, fit_values, imputer_value):
        for j in range(len(data)):
            for i in range(len(data[j])):
                if data[j][i].lower() in imputer_value:
                    data[j][i] = str(fit_values[i])
        return data


    def fit_test_data_float(self, data, fit_values, imputer_value):
        for j in range(len(data)):
            for i in range(len(data[j])):
                if data[j][i].lower() in imputer_value:
                    data[j][i] = float(fit_values[i])
                data[j][i] = float(data[j][i])
        return data

    def test_fit_min(self):
        imputer = Imputer()
        process_data, cols_transform_value = imputer.fit(self.table_instance, "min", output_format = 'str')
        cols_transform_value_ground_true = [-1.451067,-2.147457,-1.166747,-1.092337,-1.037534,-1.302401,-1.305831,-1.745063,-1.591501,-1.230554]
        imputer_value = [ '', 'none', 'na', 'null' ] 
        test_data_fit = self.fit_test_data(self.test_data, cols_transform_value_ground_true, imputer_value)

        self.assertListEqual(self.table_to_list(process_data), test_data_fit)
        self.assertListEqual(cols_transform_value, cols_transform_value_ground_true)

    def test_fit_max(self):
        imputer = Imputer()
        process_data, cols_transform_value = imputer.fit(self.table_instance, "max", output_format = 'str')
        cols_transform_value_ground_true = [0.963102,1.467675,0.829202,0.772457,1.000835,0.962702,1.077099,1.053586,2.996525,0.961696]
        imputer_value = [ '', 'none', 'na', 'null' ] 
        test_data_fit = self.fit_test_data(self.test_data, cols_transform_value_ground_true, imputer_value)

        self.assertListEqual(self.table_to_list(process_data), test_data_fit)
        self.assertListEqual(cols_transform_value, cols_transform_value_ground_true)

    def test_fit_mean(self):
        imputer = Imputer()
        process_data, cols_transform_value = imputer.fit(self.table_instance, "mean", output_format = 'str')
        cols_transform_value_ground_true = [-0.413542, -0.330818, -0.343831, -0.444957, -0.107726, -0.569688, -0.548734, -0.670353, 0.002498, -0.275518]
        imputer_value = [ '', 'none', 'na', 'null' ] 
        test_data_fit = self.fit_test_data(self.test_data, cols_transform_value_ground_true, imputer_value)

        self.assertListEqual(self.table_to_list(process_data), test_data_fit)
        self.assertListEqual(cols_transform_value, cols_transform_value_ground_true)


    def test_fit_replace_value(self):
        imputer_value = [ 'NA', 'naaa' ] 
        imputer = Imputer(imputer_value)
        process_data, cols_transform_value = imputer.fit(self.table_instance,replace_method="designated", replace_value='111111', output_format = 'str')
        cols_transform_value_ground_true = [ '111111' for _ in range(10) ]
        test_data_fit = self.fit_test_data(self.test_data, cols_transform_value_ground_true, imputer_value)

        self.assertListEqual(self.table_to_list(process_data), test_data_fit)
        self.assertListEqual(cols_transform_value, cols_transform_value_ground_true)

    def test_fit_none_replace_method(self):
        imputer_value = [ 'NA', 'naaa' ] 
        imputer = Imputer(imputer_value)
        process_data, cols_transform_value = imputer.fit(self.table_instance, output_format = 'str')
        cols_transform_value_ground_true = [ '0' for _ in range(10) ]
        test_data_fit = self.fit_test_data(self.test_data, cols_transform_value_ground_true, imputer_value)

        self.assertListEqual(self.table_to_list(process_data), test_data_fit)
        self.assertListEqual(cols_transform_value, cols_transform_value_ground_true)
    
    def test_fit_max_float(self):
        imputer = Imputer()
        process_data, cols_transform_value = imputer.fit(self.table_instance, "max", output_format = 'float')
        cols_transform_value_ground_true = [0.963102,1.467675,0.829202,0.772457,1.000835,0.962702,1.077099,1.053586,2.996525,0.961696]
        imputer_value = [ '', 'none', 'na', 'null' ] 
        test_data_fit = self.fit_test_data_float(self.test_data, cols_transform_value_ground_true, imputer_value)

        self.assertListEqual(self.table_to_list(process_data), test_data_fit)
        self.assertListEqual(cols_transform_value, cols_transform_value_ground_true)

if __name__ == "__main__":
    unittest.main()
