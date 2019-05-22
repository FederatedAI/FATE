#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import numpy as np
import time
import random
import string
import unittest

from federatedml.util import DenseFeatureReader
from federatedml.util import SparseFeatureReader
from federatedml.util import SparseTagReader
from federatedml.param import DataIOParam
from arch.api import eggroll
from arch.api import federation
from arch.api import storage
from federatedml.util import consts


class TestDenseFeatureReader(unittest.TestCase):
    def setUp(self):
        self.table = "dataio_dense_table_test"
        self.namespace = "dataio_test"
        data1 = [("a", "1,2,-1,0,0,5"), ("b", "4,5,6,0,1,2")]
        save_data(data1, self.table, self.namespace)        

        self.table2 = "dataio_dense_table_test2"
        data2 = [("a", '-1,,NA,NULL,null,2')]
        save_data(data2, self.table2, self.namespace)        

    def test_dense_output_format(self):
        dataio_param = DataIOParam()
        reader = DenseFeatureReader(dataio_param)
        data = reader.read_data(self.table, self.namespace).collect()
        result = dict(data)
        self.assertTrue(type(result['a']).__name__ == "Instance")
        self.assertTrue(type(result['b']).__name__ == "Instance")
        vala = result['a']
        features = vala.features
        weight = vala.weight
        label = vala.label
        self.assertTrue(np.abs(weight - 1.0) < consts.FLOAT_ZERO)
        self.assertTrue(type(features).__name__ == "ndarray")
        self.assertTrue(label == None)
        self.assertTrue(features.shape[0] == 6)
        self.assertTrue(features.dtype == "float64")

    def test_sparse_output_format(self):
        dataio_param = DataIOParam()
        dataio_param.output_format = "sparse"
        reader = DenseFeatureReader(dataio_param)
        data = reader.read_data(self.table, self.namespace).collect()
        result = dict(data)
        vala = result['a']
        features = vala.features
        self.assertTrue(type(features).__name__ == "SparseVector")
        self.assertTrue(len(features.sparse_vec) == 4)
        self.assertTrue(features.shape == 6)

    def test_missing_value_fill(self):
        dataio_param = DataIOParam()
        dataio_param.missing_fill = True
        dataio_param.with_label = False
        dataio_param.output_format = "sparse"
        dataio_param.default_value = 100
        dataio_param.missing_fill_method = "designated"
        dataio_param.data_type = 'int'
        reader = DenseFeatureReader(dataio_param)
        data = reader.read_data(self.table2, self.namespace).collect()
        result = dict(data)
        features = result['a'].features
        for i in range(1, 5):
            self.assertTrue(features.get_data(i) == 100)

    def test_with_label(self):
        dataio_param = DataIOParam()
        dataio_param.with_label = True
        dataio_param.label_idx = 2
        reader = DenseFeatureReader(dataio_param)
        data = reader.read_data(self.table, self.namespace).collect()
        result = dict(data)
        vala = result['a']
        label = vala.label
        features = vala.features
        self.assertTrue(label == -1)
        self.assertTrue(features.shape[0] == 5)


class TestSparseFeatureReader(unittest.TestCase):
    def setUp(self):
        self.table = "dataio_sparse_table_test"
        self.namespace = "dataio_test"
        self.data = []
        self.max_feature = -1
        for i in range(100):
            row = []
            label = i % 2
            row.append(str(label))
            dict = {}
            
            for j in range(20):
                x = random.randint(0, 1000)
                val = random.random()
                if x in dict:
                    continue
                self.max_feature = max(self.max_feature, x)
                dict[x] = True
                row.append(":".join(map(str, [x, val])))
             
            self.data.append((i, " ".join(row)))

        save_data(self.data, self.table, self.namespace)        

    
    def test_sparse_output_format(self):
        dataio_param = DataIOParam()
        dataio_param.input_format = "sparse"
        dataio_param.delimitor = ' '
        dataio_param.default_value = 2**30
        dataio_param.output_format = "sparse"
        reader = SparseFeatureReader(dataio_param)
        insts = list(reader.read_data(self.table, self.namespace).collect()) 
        for i in range(100):
            self.assertTrue(insts[i][1].features.get_shape() == self.max_feature + 1)
            self.assertTrue(insts[i][1].label == i % 2)
            original_feat = {}
            row = self.data[i][1].split(" ")
            for j in range(1, len(row)):
                fid, val = row[j].split(":", -1)
                original_feat[int(fid)] = float(val)
 
            self.assertTrue(original_feat == insts[i][1].features.sparse_vec)

    def test_dense_output_format(self):
        dataio_param = DataIOParam()
        dataio_param.input_format = "sparse"
        dataio_param.delimitor = ' '
        dataio_param.output_format = "dense"
        reader = SparseFeatureReader(dataio_param)
        insts = list(reader.read_data(self.table, self.namespace).collect()) 
        for i in range(100):
            features = insts[i][1].features
            self.assertTrue(type(features).__name__ == "ndarray")
            self.assertTrue(features.shape[0] == self.max_feature + 1)
            self.assertTrue(insts[i][1].label == i % 2)
            
            row = self.data[i][1].split(" ")
            ori_feat = [0 for i in range(self.max_feature + 1)]
            for j in range(1, len(row)):
                fid, val = row[j].split(":", -1)
                ori_feat[int(fid)] = float(val)
            
            ori_feat = np.asarray(ori_feat, dtype=dataio_param.data_type)
    
            self.assertTrue(np.abs(ori_feat - features).any() < consts.FLOAT_ZERO)

    def test_dense_output_format(self):
        dataio_param = DataIOParam()
        dataio_param.input_format = "sparse"
        dataio_param.delimitor = ' '
        dataio_param.output_format = "sparse"
        reader = SparseFeatureReader(dataio_param)
        insts = list(reader.read_data(self.table, self.namespace).collect()) 
        for i in range(100):
            features = insts[i][1].features
            self.assertTrue(type(features).__name__ == "SparseVector")
            self.assertTrue(features.get_shape() == self.max_feature + 1)
            self.assertTrue(insts[i][1].label == i % 2)
            
            row = self.data[i][1].split(" ")
            for j in range(1, len(row)):
                fid, val = row[j].split(":", -1)

                self.assertTrue(np.fabs(features.get_data(int(fid)) - float(val)) < consts.FLOAT_ZERO)


class TestSparseTagReader(unittest.TestCase):
    def setUp(self):
        self.table = "dataio_sparse_tag_test"
        self.namespace = "dataio_test"
       
        self.table2 = "dataio_sparse_tag_value_test"
        
        self.data = []
        self.data_with_value = []
        for i in range(100):
            row = []
            row_with_value = []
            for j in range(100):
                if random.randint(1, 100) > 30:
                    continue
                str_r = ''.join(random.sample(string.ascii_letters + string.digits, 10))
                row.append(str_r)
                row_with_value.append(str_r + ':' + str(random.random()))

            self.data.append((i, ' '.join(row)))
            self.data_with_value.append((i, ' '.join(row_with_value)))

        save_data(self.data, self.table, self.namespace)       
        save_data(self.data_with_value, self.table2, self.namespace)

    def test_tag_sparse_output_format(self):
        dataio_param = DataIOParam()
        dataio_param.input_format = "tag"
        dataio_param.data_type = "int"
        dataio_param.delimitor = ' '
        dataio_param.with_label = False
        dataio_param.output_format = "sparse"
        reader = SparseTagReader(dataio_param)
        tag_insts = reader.read_data(self.table, self.namespace)
        features = [inst.features for key, inst in tag_insts.collect()]

        tags = set()
        for row in self.data:
            tags |= set(row[1].split(" ", -1))
   
        tags = sorted(tags)
        tag_dict = dict(zip(tags, range(len(tags))))
        
        for i in range(len(self.data)):
            ori_feature = {}
            for tag in self.data[i][1].split(" ", -1):
                ori_feature[tag_dict.get(tag)] = 1

            self.assertTrue(ori_feature == features[i].sparse_vec)

    def test_tag_with_value_sparse_output_format(self):
        dataio_param = DataIOParam()
        dataio_param.input_format = "tag"
        dataio_param.data_type = "float"
        dataio_param.tag_with_value = True
        dataio_param.tag_value_delimitor = ":"
        dataio_param.delimitor = ' '
        dataio_param.with_label = False
        dataio_param.output_format = "sparse"
        reader = SparseTagReader(dataio_param)
        tag_insts = reader.read_data(self.table2, self.namespace)
        features = [inst.features for key, inst in tag_insts.collect()]

        tags = set()
        for row in self.data_with_value:
            tag_list = []
            for tag_with_value in row[1].split(" ", -1):
                tag_list.append(tag_with_value.split(":")[0])

            tags |= set(tag_list)
   
        tags = sorted(tags)
        tag_dict = dict(zip(tags, range(len(tags))))
        
        for i in range(len(self.data_with_value)):
            ori_feature = {}
            for tag_with_value in self.data_with_value[i][1].split(" ", -1):
                idx = tag_dict.get(tag_with_value.split(":")[0])
                val = float(tag_with_value.split(":")[1])

                self.assertTrue(np.abs(val - features[i].get_data(idx)) < consts.FLOAT_ZERO) 
    
    def test_tag_dense_output_format(self):
        dataio_param = DataIOParam()
        dataio_param.input_format = "tag"
        dataio_param.data_type = 'int'
        dataio_param.delimitor = ' '
        dataio_param.with_label = False
        dataio_param.output_format = "dense"
        reader = SparseTagReader(dataio_param)
        tag_insts = reader.read_data(self.table, self.namespace)
        features = [inst.features for key, inst in tag_insts.collect()]

        tags = set()
        for row in self.data:
            tags |= set(row[1].split(" ", -1))
        
        tags = sorted(tags)
        tag_dict = dict(zip(tags, range(len(tags))))
        
        for i in range(len(self.data)):
            ori_feature = [0 for i in range(len(tags))]

            for tag in self.data[i][1].split(" ", -1):
                ori_feature[tag_dict.get(tag)] = 1
            
            ori_feature = np.asarray(ori_feature, dtype='int')
            self.assertTrue(np.abs(ori_feature - features).all() < consts.FLOAT_ZERO)

    def test_tag_with_value_dense_output_format(self):
        dataio_param = DataIOParam()
        dataio_param.input_format = "tag"
        dataio_param.data_type = 'float'
        dataio_param.delimitor = ' '
        dataio_param.with_label = False
        dataio_param.output_format = "dense"
        reader = SparseTagReader(dataio_param)
        tag_insts = reader.read_data(self.table2, self.namespace)
        features = [inst.features for key, inst in tag_insts.collect()]

        tags = set()
        for row in self.data_with_value:
            tag_list = []
            for tag_with_value in row[1].split(" ", -1):
                tag_list.append(tag_with_value.split(":")[0])

            tags |= set(tag_list)
        
        tags = sorted(tags)
        tag_dict = dict(zip(tags, range(len(tags))))
        
        for i in range(len(self.data_with_value)):
            ori_feature = [0 for i in range(len(tags))]

            for tag_with_value in self.data_with_value[i][1].split(" ", -1):
                tag = tag_with_value.split(":", -1)[0]
                val = float(tag_with_value.split(":", -1)[1])
                ori_feature[tag_dict.get(tag)] = val
            
            ori_feature = np.asarray(ori_feature, dtype='float64')
            self.assertTrue(np.abs(ori_feature - features).all() < consts.FLOAT_ZERO)


def save_data(input_data, table_name, namespace):
    storage.save_data(input_data, table_name, namespace)
   

if __name__ == '__main__':
    eggroll.init("test_dataio" + str(int(time.time())))
    federation.init("test_dataio", 
                    {"local": {
                       "role": "guest",
                       "party_id": 9999
                    },
                     "role": {
                       "host": [
                           10000
                       ],
                       "guest": [
                           9999
                       ]
                     }
                    })
    unittest.main()
