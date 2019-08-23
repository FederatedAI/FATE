# #
# #  Copyright 2019 The FATE Authors. All Rights Reserved.
# #
# #  Licensed under the Apache License, Version 2.0 (the "License");
# #  you may not use this file except in compliance with the License.
# #  You may obtain a copy of the License at
# #
# #      http://www.apache.org/licenses/LICENSE-2.0
# #
# #  Unless required by applicable law or agreed to in writing, software
# #  distributed under the License is distributed on an "AS IS" BASIS,
# #  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# #  See the License for the specific language governing permissions and
# #  limitations under the License.
# #
#
# from federatedml.secureprotol.encrypt import FakeEncrypt, PaillierEncrypt
# from federatedml.frameworks.homo.procedure.transfer import host2arbiter, arbiter2host
# from federatedml.frameworks.homo.procedure.base import Coordinate
# from federatedml.util.transfer_variable.homo_transfer_variable import HomeModelTransferVariable
#
#
# class GenPaillierCipher(Coordinate):
#     """@host -> @arbiter -> @host
#     hosts transfer a flag to arbiter, and then arbiter generate paillier public key for each host
#     """
#
#     @staticmethod
#     def from_transfer_variable(transfer_variable: HomeModelTransferVariable):
#         return GenPaillierCipher(
#             use_encrypt_transfer_variable_name=transfer_variable.use_encrypt.name,
#             use_encrypt_transfer_tag=transfer_variable.generate_transferid(transfer_variable.use_encrypt),
#             paillier_pubkey_transfer_variable_name=transfer_variable.paillier_pubkey.name,
#             paillier_pubkey_transfer_tag=transfer_variable.generate_transferid(transfer_variable.paillier_pubkey)
#         )
#
#     def __init__(self,
#                  use_encrypt_transfer_variable_name,
#                  use_encrypt_transfer_tag,
#                  paillier_pubkey_transfer_variable_name,
#                  paillier_pubkey_transfer_tag):
#         self._use_encrypt_scatter = host2arbiter(name=use_encrypt_transfer_variable_name,
#                                                  tag=use_encrypt_transfer_tag)
#         self._paillier_pubkey_broadcast = arbiter2host(name=paillier_pubkey_transfer_variable_name,
#                                                        tag=paillier_pubkey_transfer_tag)
#
#     def host_call(self, use_encryption):
#         # tell arbiter if this host use encryption or not
#         self._use_encrypt_scatter.remote(use_encryption)
#
#         # get public key from arbiter
#         pubkey = self._paillier_pubkey_broadcast.get() if use_encryption else None
#
#         return pubkey
#
#     def arbiter_call(self, key_length):
#         hosts_use_encryption = self._use_encrypt_scatter.get()
#         host_ciphers = []
#         for idx, use_encryption in enumerate(hosts_use_encryption):
#             if not use_encryption:
#                 cipher = FakeEncrypt()
#             else:
#                 cipher = PaillierEncrypt()
#                 cipher.generate_key(key_length)
#                 pub_key = cipher.get_public_key()
#                 self._paillier_pubkey_broadcast.remote(pub_key)
#             host_ciphers.append(cipher)
#         return hosts_use_encryption, host_ciphers
#
#     def guest_call(self, **kwargs):
#         raise NotImplemented(f"never calling `guest_call` in {self}")
