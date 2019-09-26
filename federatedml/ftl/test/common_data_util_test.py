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

import numpy as np

from arch.api.session import init
from federatedml.ftl.data_util.common_data_util import overlapping_samples_converter, generate_table_namespace_n_name
from federatedml.ftl.test.util import assert_matrix


def fetch_overlap_data(data_dict, overlap_indexes, nonoverlap_indexes):
    overlap_data = []
    nonoverlap_data = []
    for i in overlap_indexes:
        overlap_data.append(data_dict[i])
    for i in nonoverlap_indexes:
        nonoverlap_data.append(data_dict[i])

    overlap_data = np.array(overlap_data)
    nonoverlap_data = np.array(nonoverlap_data)
    return overlap_data, nonoverlap_data


class TestCommonDataUtil(unittest.TestCase):

    def test_generate_table_namespace_n_name(self):
        infile = "UCI_Credit_Card.csv"
        ns, name = generate_table_namespace_n_name(infile)

        infile0 = "/UCI_Credit_Card.csv"
        ns0, name0 = generate_table_namespace_n_name(infile0)

        infile1 = "../../../data/UCI_Credit_Card/UCI_Credit_Card.csv"
        ns1, name1 = generate_table_namespace_n_name(infile1)

        infile2 = "/data/projects/est/UCI_Credit_Card/UCI_Credit_Card.csv"
        ns2, name2 = generate_table_namespace_n_name(infile2)

        assert ns == ns0 == ns1 == ns2
        assert name == name0 == name1 == name2

    def test_convert_overlapping_samples_and_labels_1(self):
        host_sample_indexes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14])
        guest_sample_indexes = np.array([8, 9, 10, 11, 12])
        before_overlap_indexes = np.array([8, 9, 10])
        before_host_nonoverlap_indexes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 13, 14])

        self.__test(host_sample_indexes, guest_sample_indexes, before_overlap_indexes, before_host_nonoverlap_indexes)

    def test_convert_overlapping_samples_and_labels_2(self):
        host_sample_indexes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        guest_sample_indexes = np.array([0, 1, 2, 3, 4, 9, 10, 11, 12])
        before_overlap_indexes = np.array([0, 1, 2, 3, 4])
        before_host_nonoverlap_indexes = np.array([5, 6, 7, 8])

        self.__test(host_sample_indexes, guest_sample_indexes, before_overlap_indexes, before_host_nonoverlap_indexes)

    @staticmethod
    def __test(host_sample_indexes, guest_sample_indexes, before_overlap_indexes, before_host_nonoverlap_indexes):
        host_x_dict = {}
        host_label_dict = {}
        np.random.seed(100)
        for i in host_sample_indexes:
            host_x_dict[i] = np.random.rand(1, 3)
            host_label_dict[i] = np.random.randint(0, 2)

        overlap_samples, nonoverlap_samples = fetch_overlap_data(host_x_dict, before_overlap_indexes,
                                                                 before_host_nonoverlap_indexes)
        overlap_labels, nonoverlap_labels = fetch_overlap_data(host_label_dict, before_overlap_indexes,
                                                               before_host_nonoverlap_indexes)

        overlap_samples = np.squeeze(overlap_samples)
        nonoverlap_samples = np.squeeze(nonoverlap_samples)
        overlap_labels = np.expand_dims(overlap_labels, axis=1)
        nonoverlap_labels = np.expand_dims(nonoverlap_labels, axis=1)

        host_x, overlap_indexes, non_overlap_indexes, host_label = overlapping_samples_converter(host_x_dict,
                                                                                                 host_sample_indexes,
                                                                                                 guest_sample_indexes,
                                                                                                 host_label_dict)

        after_conversion_overlap_samples = host_x[overlap_indexes]
        after_conversion_nonoverlap_samples = host_x[non_overlap_indexes]
        after_conversion_overlap_labels = host_label[overlap_indexes]
        after_conversion_nonoverlap_labels = host_label[non_overlap_indexes]

        assert_matrix(overlap_samples, after_conversion_overlap_samples)
        assert_matrix(nonoverlap_samples, after_conversion_nonoverlap_samples)
        assert_matrix(overlap_labels, after_conversion_overlap_labels)
        assert_matrix(nonoverlap_labels, after_conversion_nonoverlap_labels)


if __name__ == '__main__':
    init()
    unittest.main()
