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
from federatedml.ftl.encryption import encryption
from federatedml.ftl.faster_encrypted_ftl import FasterEncryptedFTLGuestModel, FasterEncryptedFTLHostModel
from federatedml.ftl.test.mock_models import MockAutoencoder, MockFTLModelParam
from federatedml.secureprotol.encrypt import PaillierEncrypt


def run_one_party_msg_exchange(autoencoderA, autoencoderB, U_A, U_B, y, overlap_indexes, non_overlap_indexes,
                               public_key=None, private_key=None):

    fake_model_param = MockFTLModelParam(alpha=1)

    partyA = FasterEncryptedFTLGuestModel(autoencoderA, fake_model_param, public_key=public_key, private_key=private_key)
    partyA.set_batch(U_A, y, non_overlap_indexes, overlap_indexes)
    partyB = FasterEncryptedFTLHostModel(autoencoderB, fake_model_param, public_key=public_key, private_key=private_key)
    partyB.set_batch(U_B, overlap_indexes)

    [y_overlap_phi, mapping_comp_A, phi, phi_2] = partyA.send_components()
    [uB_overlap, mapping_comp_B] = partyB.send_components()

    partyA.receive_components([uB_overlap, mapping_comp_B])
    partyB.receive_components([y_overlap_phi, mapping_comp_A, phi, phi_2])

    precomputed_components_A = partyA.send_precomputed_components()
    precomputed_components_B = partyB.send_precomputed_components()

    partyA.receive_precomputed_components(precomputed_components_B)
    partyB.receive_precomputed_components(precomputed_components_A)

    # encrypt_gradients_A = partyA.send_gradients()
    # encrypt_gradients_B = partyB.send_gradients()

    # partyA.receive_gradients(__decrypt_gradients(encrypt_gradients_A))
    # partyB.receive_gradients(__decrypt_gradients(encrypt_gradients_B))

    return partyA, partyB


class TestFasterEncryptedGradients(unittest.TestCase):

    def setUp(self):
        paillierEncrypt = PaillierEncrypt()
        paillierEncrypt.generate_key()
        self.public_key = paillierEncrypt.get_public_key()
        self.private_key = paillierEncrypt.get_privacy_key()

    def test_party_b_gradient_checking_test(self):

        U_A = np.array([[1, 2, 3, 4, 5],
                        [4, 5, 6, 7, 8],
                        [7, 8, 9, 10, 11],
                        [4, 5, 6, 7, 8]])
        U_B = np.array([[4, 2, 3, 1, 2],
                        [6, 5, 1, 4, 5],
                        [7, 4, 1, 9, 10],
                        [6, 5, 1, 4, 5]])
        y = np.array([[1], [-1], [1], [-1]])

        overlap_indexes = [1, 2]
        non_overlap_indexes = [0, 3]

        Wh = np.ones((4, U_A.shape[1]))
        bh = np.zeros(U_A.shape[1])

        autoencoderA = MockAutoencoder(0)
        autoencoderA.build(U_A.shape[1], Wh, bh)
        autoencoderB = MockAutoencoder(1)
        autoencoderB.build(U_B.shape[1], Wh, bh)

        partyA, partyB = run_one_party_msg_exchange(autoencoderA, autoencoderB, U_A, U_B, y, overlap_indexes,
                                                    non_overlap_indexes, self.public_key, self.private_key)
        loss_grads_B_1 = partyB.get_loss_grads()
        loss1 = partyA.send_loss()

        U_B_prime = np.array([[4, 2, 3, 1, 2],
                              [6, 5, 1.001, 4, 5],
                              [7, 4, 1, 9, 10],
                              [6, 5, 1, 4, 5]])
        partyA, partyB = run_one_party_msg_exchange(autoencoderA, autoencoderB, U_A, U_B_prime, y, overlap_indexes,
                                                    non_overlap_indexes, self.public_key, self.private_key)
        loss_grads_B_2 = partyB.get_loss_grads()
        loss2 = partyA.send_loss()

        loss_grads_B_1 = np.array(encryption.decrypt_matrix(self.private_key, loss_grads_B_1))
        loss_grads_B_2 = np.array(encryption.decrypt_matrix(self.private_key, loss_grads_B_2))

        loss1 = encryption.decrypt(self.private_key, loss1)
        loss2 = encryption.decrypt(self.private_key, loss2)

        grad_approx = (loss2 - loss1) / 0.001
        grad_real = loss_grads_B_1[0, 2]
        grad_diff = np.abs(grad_approx - grad_real)
        assert grad_diff < 0.001

    def test_party_a_gradient_checking_test(self):

        U_A = np.array([[1, 2, 3, 4, 5],
                        [4, 5, 6, 7, 8],
                        [7, 8, 9, 10, 11],
                        [4, 5, 6, 7, 8]])
        U_B = np.array([[4, 2, 3, 1, 2],
                        [6, 5, 1, 4, 5],
                        [7, 4, 1, 9, 10],
                        [6, 5, 1, 4, 5]])
        y = np.array([[1], [-1], [1], [-1]])

        overlap_indexes = [1, 2]
        non_overlap_indexes = [0, 3]

        Wh = np.ones((4, U_A.shape[1]))
        bh = np.zeros(U_A.shape[1])

        autoencoderA = MockAutoencoder(0)
        autoencoderA.build(U_A.shape[1], Wh, bh)
        autoencoderB = MockAutoencoder(1)
        autoencoderB.build(U_B.shape[1], Wh, bh)

        partyA, _ = run_one_party_msg_exchange(autoencoderA, autoencoderB, U_A, U_B, y, overlap_indexes,
                                               non_overlap_indexes, self.public_key, self.private_key)
        loss_grads_A_1 = partyA.get_loss_grads()
        loss1 = partyA.send_loss()

        U_A_prime = np.array([[1, 2, 3, 4, 5],
                              [4, 5.001, 6, 7, 8],
                              [7, 8, 9, 10, 11],
                              [4, 5, 6, 7, 8]])
        partyA, _ = run_one_party_msg_exchange(autoencoderA, autoencoderB, U_A_prime, U_B, y, overlap_indexes,
                                               non_overlap_indexes, self.public_key, self.private_key)
        loss_grads_A_2 = partyA.get_loss_grads()
        loss2 = partyA.send_loss()

        loss_grads_A_1 = np.array(encryption.decrypt_matrix(self.private_key, loss_grads_A_1))
        loss_grads_A_2 = np.array(encryption.decrypt_matrix(self.private_key, loss_grads_A_2))

        loss1 = encryption.decrypt(self.private_key, loss1)
        loss2 = encryption.decrypt(self.private_key, loss2)

        grad_approx = (loss2 - loss1) / 0.001
        grad_real = loss_grads_A_1[1, 1]
        grad_diff = np.abs(grad_approx - grad_real)
        assert grad_diff < 0.001


if __name__ == '__main__':
    init()
    unittest.main()
