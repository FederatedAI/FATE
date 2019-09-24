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

from federatedml.ftl.hetero_ftl.hetero_ftl_guest import GuestFactory, HeteroEncryptFTLGuest, \
    FasterHeteroEncryptFTLGuest, HeteroPlainFTLGuest
from federatedml.ftl.hetero_ftl.hetero_ftl_host import HostFactory, HeteroEncryptFTLHost, \
    FasterHeteroEncryptFTLHost, HeteroPlainFTLHost
from federatedml.ftl.test.mock_models import MockAutoencoder
from federatedml.param.ftl_param import FTLModelParam
from federatedml.transfer_variable.transfer_class.hetero_ftl_transfer_variable import HeteroFTLTransferVariable


class TestHostGuestConstructor(unittest.TestCase):

    def test_create_enc_ftl_host(self):
        ftl_model_param = FTLModelParam(is_encrypt=True, enc_ftl=None)
        host = self.create_host(ftl_model_param)
        self.assertTrue(isinstance(host, HeteroEncryptFTLHost))

    def test_create_enc_faster_ftl_host(self):
        ftl_model_param = FTLModelParam(is_encrypt=True, enc_ftl="enc_ftl2")
        host = self.create_host(ftl_model_param)
        self.assertTrue(isinstance(host, FasterHeteroEncryptFTLHost))

    def test_create_plain_ftl_host(self):
        ftl_model_param = FTLModelParam(is_encrypt=False)
        host = self.create_host(ftl_model_param)
        self.assertTrue(isinstance(host, HeteroPlainFTLHost))

    def test_create_enc_ftl_guest(self):
        ftl_model_param = FTLModelParam(is_encrypt=True, enc_ftl=None)
        guest = self.create_guest(ftl_model_param)
        self.assertTrue(isinstance(guest, HeteroEncryptFTLGuest))

    def test_create_enc_faster_ftl_guest(self):
        ftl_model_param = FTLModelParam(is_encrypt=True, enc_ftl="enc_ftl2")
        guest = self.create_guest(ftl_model_param)
        self.assertTrue(isinstance(guest, FasterHeteroEncryptFTLGuest))

    def test_create_plain_ftl_guest(self):
        ftl_model_param = FTLModelParam(is_encrypt=False)
        guest = self.create_guest(ftl_model_param)
        self.assertTrue(isinstance(guest, HeteroPlainFTLGuest))

    @staticmethod
    def create_host(ftl_model_param):
        transfer_variable = HeteroFTLTransferVariable()
        Wh = np.ones((4, 2))
        bh = np.zeros(2)
        ftl_local_model = MockAutoencoder("01")
        ftl_local_model.build(2, Wh, bh)
        return HostFactory.create(ftl_model_param, transfer_variable, ftl_local_model)

    @staticmethod
    def create_guest(ftl_model_param):
        transfer_variable = HeteroFTLTransferVariable()
        Wh = np.ones((4, 2))
        bh = np.zeros(2)
        ftl_local_model = MockAutoencoder("02")
        ftl_local_model.build(2, Wh, bh)
        return GuestFactory.create(ftl_model_param, transfer_variable, ftl_local_model)


if __name__ == '__main__':
    unittest.main()
