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
from federatedml.feature.instance import Instance
from federatedml.ftl.data_util.common_data_util import create_table
from federatedml.ftl.hetero_ftl.hetero_ftl_guest import HeteroPlainFTLGuest
from federatedml.ftl.plain_ftl import PlainFTLGuestModel
from federatedml.ftl.plain_ftl import PlainFTLHostModel
from federatedml.ftl.test.mock_models import MockAutoencoder, MockDiffConverge
from federatedml.param.param import FTLModelParam
from federatedml.transfer_variable.transfer_class.hetero_ftl_transfer_variable import HeteroFTLTransferVariable


class TestHeteroFTLGuest(HeteroPlainFTLGuest):

    def __init__(self, guest, model_param, transfer_variable):
        super(TestHeteroFTLGuest, self).__init__(guest, model_param, transfer_variable)

        U_B = np.array([[4, 2, 3, 1, 2],
                        [6, 5, 1, 4, 5],
                        [7, 4, 1, 9, 10],
                        [6, 5, 1, 4, 5]])

        overlap_indexes = [1, 2]

        Wh = np.ones((5, U_B.shape[1]))
        bh = np.zeros(U_B.shape[1])

        autoencoderB = MockAutoencoder(1)
        autoencoderB.build(U_B.shape[1], Wh, bh)

        self.host = PlainFTLHostModel(autoencoderB, self.model_param)
        self.host.set_batch(U_B, overlap_indexes)

    def _do_remote(self, value=None, name=None, tag=None, role=None, idx=None):
        print("@_do_remote", value, name, tag, role, idx)

    def _do_get(self, name=None, tag=None, idx=None):
        print("@_do_get", name, tag, idx)
        if tag == "HeteroFTLTransferVariable.host_sample_indexes.0":
            return [np.array([1, 2, 4, 5])]
        elif tag == "HeteroFTLTransferVariable.host_component_list.0.0":
            return self.host.send_components()
        return None


class TestCreateGuestHostEggrollTable(unittest.TestCase):

    def test_hetero_plain_guest_prepare_table(self):
        U_A = np.array([[1, 2, 3, 4, 5],
                        [4, 5, 6, 7, 8],
                        [7, 8, 9, 10, 11],
                        [4, 5, 6, 7, 8]])
        y = np.array([[1], [-1], [1], [-1]])

        Wh = np.ones((5, U_A.shape[1]))
        bh = np.zeros(U_A.shape[1])

        model_param = FTLModelParam(alpha=1, max_iteration=1)

        autoencoderA = MockAutoencoder(0)
        autoencoderA.build(U_A.shape[1], Wh, bh)
        guest = PlainFTLGuestModel(autoencoderA, model_param)

        converge_func = MockDiffConverge(None)
        ftl_guest = TestHeteroFTLGuest(guest, model_param, HeteroFTLTransferVariable())
        ftl_guest.set_converge_function(converge_func)

        guest_sample_indexes = np.array([0, 1, 2, 3])
        guest_x_dict = {}
        guest_label_dict = {}
        instance_dict = {}
        instance_list = []
        np.random.seed(100)
        for i, feature, label, in zip(guest_sample_indexes, U_A, y):
            instance = Instance(inst_id=i, features=feature, label=label[0])
            guest_x_dict[i] = feature
            guest_label_dict[i] = label[0]
            instance_dict[i] = instance
            instance_list.append(instance)

        guest_x = create_table(instance_list, indexes=guest_sample_indexes)

        guest_x, overlap_indexes, non_overlap_indexes, guest_y = ftl_guest.prepare_data(guest_x)
        print("guest_x", guest_x)
        print("overlap_indexes", overlap_indexes)
        print("non_overlap_indexes", non_overlap_indexes)
        print("guest_y", guest_y)


if __name__ == '__main__':
    init()
    unittest.main()
