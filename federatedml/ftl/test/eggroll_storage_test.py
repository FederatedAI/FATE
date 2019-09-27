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

import unittest
import uuid

import numpy as np

from arch.api.session import init, table
from federatedml.ftl.data_util.common_data_util import save_data_to_eggroll_table, create_guest_host_data_generator, \
    create_table, split_into_guest_host_dtable, load_model_parameters, save_model_parameters, feed_into_dtable
from federatedml.ftl.test.util import assert_array, assert_matrix


class TestEggrollStorage(unittest.TestCase):

    def test_create_table_with_array_1(self):

        row_count = 10
        expect_data = np.random.rand(row_count, 10)
        actual_data = np.zeros((row_count, 10))
        dtable = create_table(expect_data)
        for item in dtable.collect():
            actual_data[item[0]] = item[1]

        assert dtable.count() == row_count
        assert_matrix(expect_data, actual_data)

    def test_create_table_with_array_2(self):

        feature_count = 10
        expect_data = np.random.rand(feature_count)
        actual_data = np.zeros((feature_count, 1))
        dtable = create_table(expect_data)
        for item in dtable.collect():
            actual_data[item[0]] = item[1]

        assert dtable.count() == feature_count
        assert_matrix(np.expand_dims(expect_data, axis=1), actual_data)

    def test_create_table_with_dict(self):

        row_count = 10
        expect_data = np.random.rand(row_count, 10)
        indexes = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
        actual_data = {}
        dtable = create_table(expect_data, indexes)
        for item in dtable.collect():
            actual_data[item[0]] = item[1]

        assert dtable.count() == len(indexes)
        for i, index in enumerate(indexes):
            assert_array(actual_data[indexes[i]], expect_data[i])

    def test_split_into_guest_host_dtable(self):

        X = np.random.rand(600, 3)
        y = np.random.rand(600, 1)
        overlap_ratio = 0.2
        guest_split_ratio = 0.5
        guest_feature_num = 16

        data_size = X.shape[0]
        overlap_size = int(data_size * overlap_ratio)
        overlap_indexes = np.array(range(overlap_size))
        particular_guest_size = int((data_size - overlap_size) * guest_split_ratio)

        expected_guest_size = overlap_size + particular_guest_size
        expected_host_size = overlap_size + data_size - expected_guest_size

        guest_data, host_data, overlap_indexes = split_into_guest_host_dtable(X, y, overlap_ratio=overlap_ratio,
                                                                              guest_split_ratio=guest_split_ratio,
                                                                              guest_feature_num=guest_feature_num)

        actual_guest_size = guest_data.count()
        actual_host_size = host_data.count()
        assert expected_guest_size == actual_guest_size
        assert expected_host_size == actual_host_size

    def test_feed_into_dtable(self):

        ids = list(range(50))
        X = np.random.rand(50, 6)
        y = np.random.rand(50, 1)
        sample_range = (10, 30)
        feature_range = (2, 5)
        expected_sample_number = sample_range[1] - sample_range[0]

        expected_ids = ids[sample_range[0]: sample_range[1]]
        expected_X = X[sample_range[0]:sample_range[1], feature_range[0]: feature_range[1]]
        expected_y = y[sample_range[0]:sample_range[1]]

        expected_data = {}
        for i, id in enumerate(expected_ids):
            expected_data[id] = {
                "X": expected_X[i],
                "y": expected_y[i]
            }

        data_table = feed_into_dtable(ids, X, y, sample_range, feature_range)

        val = data_table.collect()
        data_dict = dict(val)

        actual_table_size = len(data_dict)
        assert expected_sample_number == actual_table_size
        for item in data_dict.items():
            id = item[0]
            inst = item[1]
            expected_item = expected_data[id]
            X_i = expected_item["X"]
            y_i = expected_item["y"]

            features = inst.features
            label = inst.label
            assert_array(X_i, features)
            assert y_i[0] == label

    def test_create_n_guest_generators(self):

        X = np.random.rand(600, 33)
        y = np.random.rand(600, 1)
        overlap_ratio = 0.2
        guest_split_ratio = 0.3
        guest_feature_num = 16

        data_size = X.shape[0]
        overlap_size = int(data_size * overlap_ratio)
        expected_overlap_indexes = np.array(range(overlap_size))
        particular_guest_size = int((data_size - overlap_size) * guest_split_ratio)

        expected_guest_size = overlap_size + particular_guest_size
        expected_host_size = overlap_size + data_size - expected_guest_size

        guest_data_generator, host_data_generator, overlap_indexes = \
            create_guest_host_data_generator(X, y,
                                             overlap_ratio=overlap_ratio,
                                             guest_split_ratio=guest_split_ratio,
                                             guest_feature_num=guest_feature_num)

        guest_features_dict = {}
        guest_labels_dict = {}
        guest_instances_indexes = []
        guest_count = 0
        guest_feature_num = 0
        for item in guest_data_generator:
            key = item[0]
            instance = item[1]
            guest_feature_num = instance.features.shape[-1]
            guest_count += 1
            guest_instances_indexes.append(key)
            guest_features_dict[key] = instance.features
            guest_labels_dict[key] = instance.label

        host_features_dict = {}
        host_labels_dict = {}
        host_instances_indexes = []
        host_count = 0
        host_feature_num = 0
        for item in host_data_generator:
            key = item[0]
            instance = item[1]
            host_feature_num = instance.features.shape[-1]
            host_count += 1
            host_instances_indexes.append(key)
            host_features_dict[key] = instance.features
            host_labels_dict[key] = instance.label

        assert_array(expected_overlap_indexes, overlap_indexes)
        assert len(expected_overlap_indexes) == len(overlap_indexes)
        assert X.shape[-1] == guest_feature_num + host_feature_num
        assert expected_guest_size == guest_count
        assert expected_host_size == host_count

        for index in overlap_indexes:
            assert guest_labels_dict[index] == host_labels_dict[index]
            assert guest_labels_dict[index] == y[index]
            assert_matrix(guest_features_dict[index], X[index, :guest_feature_num].reshape(1, -1))
            assert_matrix(host_features_dict[index], X[index, guest_feature_num:].reshape(1, -1))

    def test_save_data_to_eggroll_table(self):

        data = [(1, 111), (3, 333), (4, 444), (6, 666)]
        namespace = str(uuid.uuid1())
        table_name = "table_name"
        save_data_to_eggroll_table(data, namespace, table_name, partition=1)

        actual_data_table = table(table_name, namespace)

        actual_data_dict = {}
        for item in actual_data_table.collect():
            actual_data_dict[item[0]] = item[1]

        assert len(data) == len(actual_data_dict)

        for item in data:
            assert item[1] == actual_data_dict[item[0]]

    def test_save_data_generator_to_eggroll_table(self):

        def data_generator(data_list):
            for elem in data_list:
                yield elem

        data = [(1, 111), (3, 333), (4, 444), (6, 666)]
        data_gen = data_generator(data)
        namespace = str(uuid.uuid1())
        table_name = "table_name"
        save_data_to_eggroll_table(data_gen, namespace, table_name, partition=1)

        actual_data_table = table(table_name, namespace)

        actual_data_dict = {}
        for item in actual_data_table.collect():
            actual_data_dict[item[0]] = item[1]

        assert len(data) == len(actual_data_dict)

        for item in data:
            assert item[1] == actual_data_dict[item[0]]

    def test_save_data_to_same_eggroll_table(self):

        data = [(1, 111), (3, 333), (4, 444), (6, 666)]
        namespace = str(uuid.uuid1())
        table_name = "table_name"
        save_data_to_eggroll_table(data, namespace, table_name, partition=1)

        save_data_to_eggroll_table(data, namespace, table_name, partition=1)

    def test_read_guest_host_eggroll_table(self):

        X = np.random.rand(30, 3)
        y = np.random.rand(30, 1)
        overlap_ratio = 0.2
        guest_split_ratio = 0.5
        guest_feature_num = 16

        tables_name = {}
        tables_name["guest_table_ns"] = "guest_table_ns_01"
        tables_name["guest_table_name"] = "guest_table_name_01"
        tables_name["host_table_ns"] = "host_table_ns_01"
        tables_name["host_table_name"] = "host_table_name_01"

        guest_data, host_data, overlap_indexes = split_into_guest_host_dtable(X, y, overlap_ratio=overlap_ratio,
                                                                              guest_split_ratio=guest_split_ratio,
                                                                              guest_feature_num=guest_feature_num,
                                                                              tables_name=tables_name)

        expected_guest_size = guest_data.count()
        expected_host_size = host_data.count()

        actual_guest_table = table(tables_name["guest_table_name"], tables_name["guest_table_ns"])
        actual_host_table = table(tables_name["host_table_name"], tables_name["host_table_ns"])

        actual_guest_size = actual_guest_table.count()
        actual_host_size = actual_host_table.count()
        assert expected_guest_size == actual_guest_size
        assert expected_host_size == actual_host_size

    def test_save_n_load_model_parameters(self):
        model_parameters = {"Wh": np.array([[1, 2, 3, 4],
                                            [5, 6, 7, 8],
                                            [9, 10, 11, 12]]),
                            "bh": np.array([1, 1, 1, 1]),
                            "Wo": np.array([[1, 2, 3, 4],
                                            [5, 6, 7, 8],
                                            [9, 10, 11, 12]]),
                            "bo": np.array([0, 0, 0, 0]),
                            "model_meta": {"learning_rate": 0.01,
                                           "input_dim": 100,
                                           "hidden_dim": 64}
                            }
        model_table_name = "table_name_" + str(uuid.uuid1())
        model_table_ns = "table_ns_" + str(uuid.uuid1())
        save_model_parameters(model_parameters, model_table_name, model_table_ns)

        actual_model_parameters = load_model_parameters(model_table_name, model_table_ns)

        assert len(model_parameters) == len(actual_model_parameters)
        for k in actual_model_parameters.keys():
            if k == "model_meta":
                print(actual_model_parameters[k])
            else:
                assert_matrix(actual_model_parameters[k], model_parameters[k])

    def test_destroy_table(self):

        row_count = 10
        expect_data = np.random.rand(row_count, 10)

        table_name = "table_name"
        table_ns = "table_ns"
        dtable = create_table(expect_data, model_table_name=table_name, model_namespace=table_ns, persistent=True)
        dtable_2 = table(name=table_name, namespace=table_ns)
        assert dtable.count() == dtable_2.count()

        dtable_2.destroy()
        dtable_3 = table(name=table_name, namespace=table_ns)
        assert dtable_3.count() == 0


if __name__ == '__main__':
    init()
    unittest.main()
