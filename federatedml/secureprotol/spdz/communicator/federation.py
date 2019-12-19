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
from federatedml.transfer_variable.transfer_class.secret_share_transfer_variable import SecretShareTransferVariable


class Communicator(object):

    def __init__(self, local_party=None, all_parties=None):
        self._transfer_variable = SecretShareTransferVariable()
        self._share_variable = self._transfer_variable.share.disable_auto_clean()
        self._rescontruct_variable = self._transfer_variable.rescontruct.set_preserve_num(3)
        self._mul_triplets_encrypted_variable = self._transfer_variable.multiply_triplets_encrypted.set_preserve_num(3)
        self._mul_triplets_cross_variable = self._transfer_variable.multiply_triplets_cross.set_preserve_num(3)

        self._local_party = self._transfer_variable.local_party() if local_party is None else local_party
        self._all_parties = self._transfer_variable.all_parties() if all_parties is None else all_parties
        self._party_idx = self._all_parties.index(self._local_party)
        self._other_parties = self._all_parties[:self._party_idx] + self._all_parties[(self._party_idx + 1):]

    @property
    def party(self):
        return self._local_party

    @property
    def parties(self):
        return self._all_parties

    @property
    def other_parties(self):
        return self._other_parties

    @property
    def party_idx(self):
        return self._party_idx

    def get_rescontruct_shares(self, tensor_name):
        return self._rescontruct_variable.get_parties(self._other_parties, suffix=(tensor_name,))

    def broadcast_rescontruct_share(self, share, tensor_name):
        return self._rescontruct_variable.remote_parties(share, self._other_parties, suffix=(tensor_name,))

    def remote_share(self, share, tensor_name, party):
        return self._share_variable.remote_parties(share, party, suffix=(tensor_name,))

    def get_share(self, tensor_name, party):
        return self._share_variable.get_parties(party, suffix=(tensor_name,))

    def remote_encrypted_tensor(self, encrypted, tag):
        return self._mul_triplets_encrypted_variable.remote_parties(encrypted, parties=self._other_parties, suffix=tag)

    def remote_encrypted_cross_tensor(self, encrypted, parties, tag):
        return self._mul_triplets_cross_variable.remote_parties(encrypted, parties=parties, suffix=tag)

    def get_encrypted_tensors(self, tag):
        return (self._other_parties,
                self._mul_triplets_encrypted_variable.get_parties(parties=self._other_parties, suffix=tag))

    def get_encrypted_cross_tensors(self, tag):
        return self._mul_triplets_cross_variable.get_parties(parties=self._other_parties, suffix=tag)

    def clean(self):
        self._rescontruct_variable.clean()
        self._share_variable.clean()
        self._rescontruct_variable.clean()
        self._mul_triplets_encrypted_variable.clean()
        self._mul_triplets_cross_variable.clean()
