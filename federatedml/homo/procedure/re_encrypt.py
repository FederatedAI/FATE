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
# import numpy as np
#
# from arch.api.utils.log_utils import LoggerFactory
# from federatedml.frameworks.homo.procedure.transfer import arbiter2host
# from federatedml.frameworks.homo.procedure.transfer import host2arbiter
#
# LOGGER = LoggerFactory.get_logger()
#
#
# def _tag_suffix(epoch_iter, batch_iter):
#     return f"{epoch_iter}.{batch_iter}"
#
#
# class ReEncrypt(object):
#     """
#         If use paillier encrypt, model weight need to be re-encrypt every several batches.
#     """
#
#     @staticmethod
#     def from_transfer_variable(transfer_variable):
#         return ReEncrypt(
#             re_encrypt_times_name=transfer_variable.re_encrypt_times.name,
#             re_encrypt_times_tag=transfer_variable.generate_transferid(transfer_variable.re_encrypt_times),
#             to_encrypt_model_name=transfer_variable.to_encrypt_model.name,
#             to_encrypt_model_tag=transfer_variable.generate_transferid(transfer_variable.to_encrypt_model),
#             re_encrypted_model_name=transfer_variable.re_encrypt_model.name,
#             re_encrypted_model_tag=transfer_variable.generate_transferid(transfer_variable.re_encrypt_model)
#         )
#
#     def __init__(self,
#                  re_encrypt_times_name,
#                  re_encrypt_times_tag,
#                  to_encrypt_model_name,
#                  to_encrypt_model_tag,
#                  re_encrypted_model_name,
#                  re_encrypted_model_tag):
#         self.h2a_model_to_re_encrypt = host2arbiter(name=to_encrypt_model_name,
#                                                     tag=to_encrypt_model_tag)
#         self.a2h_model_re_encrypted = arbiter2host(name=re_encrypted_model_name,
#                                                    tag=re_encrypted_model_tag)
#
#         self.h2a_re_encrypt_times = host2arbiter(name=re_encrypt_times_name, tag=re_encrypt_times_tag)
#
#         # arbiter only
#         self.re_encrypt_times = None
#
#     def arbiter_create_re_encrypt(self, host_use_cipher):
#         self.re_encrypt_times = [0] * len(host_use_cipher)
#         for idx, use_encryption in enumerate(host_use_cipher):
#             if use_encryption:
#                 self.re_encrypt_times[idx] = self.h2a_re_encrypt_times.get(idx=idx)
#         LOGGER.info("re encrypt times for all parties: {}".format(self.re_encrypt_times))
#
#     def host_create_re_encrypt(self, enable_cipher, re_encrypt_times):
#         if enable_cipher:
#             self.h2a_re_encrypt_times.remote(re_encrypt_times, idx=0)
#             LOGGER.info("sent re_encrypt_times: {}".format(re_encrypt_times))
#
#     def arbiter_re_encrypt(self, iter_num, re_encrypt_batches, host_cipher):
#         left_re_encrypt_times = self.re_encrypt_times.copy()
#         total = sum(left_re_encrypt_times)
#         batch_iter_num = 0
#         while total > 0:
#             batch_iter_num += re_encrypt_batches
#             for idx, left_times in enumerate(left_re_encrypt_times):
#                 if left_times > 0:
#                     re_encrypt_model = self.h2a_model_to_re_encrypt.get(idx=idx,
#                                                                         suffix=_tag_suffix(iter_num, batch_iter_num))
#                     cipher = host_cipher[idx]
#                     decrypt_model = cipher.decrypt_list(re_encrypt_model)
#                     re_encrypt_model = cipher.encrypt_list(decrypt_model)
#                     self.a2h_model_re_encrypted.remote(value=re_encrypt_model,
#                                                        idx=idx,
#                                                        suffix=_tag_suffix(iter_num, batch_iter_num))
#                     left_times -= 1
#                     left_re_encrypt_times[idx] = left_times
#                     total -= 1
#
#     def host_re_encrypt(self, w, iter_num, batch_num):
#         self.h2a_model_to_re_encrypt.remote(value=w, suffix=_tag_suffix(iter_num, batch_num))
#         _w = self.a2h_model_re_encrypted.get(idx=0, suffix=_tag_suffix(iter_num, batch_num))
#         return np.array(_w)
