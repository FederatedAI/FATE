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

import time

import numpy as np

from arch.api.utils import log_utils
from federatedml.evaluation import Evaluation
from federatedml.ftl.data_util.common_data_util import overlapping_samples_converter, load_model_parameters, \
    save_model_parameters, create_table, convert_instance_table_to_dict, convert_instance_table_to_array, \
    add_random_mask_for_list_of_values, remove_random_mask_from_list_of_values
from federatedml.ftl.data_util.log_util import create_shape_msg
from federatedml.ftl.eggroll_computation.helper import distribute_decrypt_matrix
from federatedml.ftl.encrypted_ftl import EncryptedFTLHostModel
from federatedml.ftl.encryption.encryption import generate_encryption_key_pair, decrypt_scalar, decrypt_array
from federatedml.ftl.faster_encrypted_ftl import FasterEncryptedFTLHostModel
from federatedml.ftl.hetero_ftl.hetero_ftl_base import HeteroFTLParty
from federatedml.ftl.plain_ftl import PlainFTLHostModel
from federatedml.param.evaluation_param import EvaluateParam
from federatedml.param.ftl_param import FTLModelParam
from federatedml.util import consts
from federatedml.transfer_variable.transfer_class.hetero_ftl_transfer_variable import HeteroFTLTransferVariable

LOGGER = log_utils.getLogger()


class HeteroFTLHost(HeteroFTLParty):

    def __init__(self, host: PlainFTLHostModel, model_param: FTLModelParam, transfer_variable: HeteroFTLTransferVariable):
        super(HeteroFTLHost, self).__init__()
        self.host_model = host
        self.model_param = model_param
        self.transfer_variable = transfer_variable
        self.max_iter = model_param.max_iter
        self.n_iter_ = 0

    def prepare_data(self, host_data):
        LOGGER.info("@ start host prepare data")
        host_features_dict, _, host_sample_indexes = convert_instance_table_to_dict(host_data)
        host_sample_indexes = np.array(host_sample_indexes)

        self._do_remote(host_sample_indexes,
                        name=self.transfer_variable.host_sample_indexes.name,
                        tag=self.transfer_variable.generate_transferid(self.transfer_variable.host_sample_indexes),
                        role=consts.GUEST,
                        idx=-1)

        guest_sample_indexes = self._do_get(name=self.transfer_variable.guest_sample_indexes.name,
                                            tag=self.transfer_variable.generate_transferid(self.transfer_variable.guest_sample_indexes),
                                            idx=-1)[0]

        host_features, overlap_indexes, _ = overlapping_samples_converter(host_features_dict, host_sample_indexes,
                                                                          guest_sample_indexes)
        return host_features, overlap_indexes

    def classified(self, prob_table, threshold):
        """
        convert a probability table into a predicted class table.
        """
        predict_table = prob_table.mapValues(lambda x: 1 if x > threshold else 0)
        return predict_table

    def evaluate(self, labels, pred_prob, pred_labels, evaluate_param: EvaluateParam):
        LOGGER.info("@ start host evaluate")
        eva = Evaluation()
        predict_res = None
        if evaluate_param.eval_type == consts.BINARY:
            eva.eval_type = consts.BINARY
            predict_res = pred_prob
        elif evaluate_param.eval_type == consts.MULTY:
            eva.eval_type = consts.MULTY
            predict_res = pred_labels
        else:
            LOGGER.warning("unknown classification type, return None as evaluation results")

        eva.pos_label = evaluate_param.pos_label
        precision_res, cuts, thresholds = eva.precision(labels=labels, pred_scores=predict_res)

        LOGGER.info("@ evaluation report:" + str(precision_res))
        return precision_res

    def predict(self, host_data, predict_param):
        LOGGER.info("@ start host predict")
        features, labels, instances_indexes = convert_instance_table_to_array(host_data)
        host_x = np.squeeze(features)
        LOGGER.debug("host_x： " + str(host_x.shape))

        host_prob = self.host_model.predict(host_x)
        self._do_remote(host_prob,
                        name=self.transfer_variable.host_prob.name,
                        tag=self.transfer_variable.generate_transferid(
                            self.transfer_variable.host_prob),
                        role=consts.GUEST, idx=-1)

        pred_prob = self._do_get(name=self.transfer_variable.pred_prob.name,
                                 tag=self.transfer_variable.generate_transferid(self.transfer_variable.pred_prob),
                                 idx=-1)[0]

        pred_prob = np.squeeze(pred_prob)
        LOGGER.debug("pred_prob: " + str(pred_prob.shape))

        pred_prob_table = create_table(pred_prob, instances_indexes)
        actual_label_table = create_table(labels, instances_indexes)
        pred_label_table = self.classified(pred_prob_table, predict_param.threshold)
        if predict_param.with_proba:
            predict_result = actual_label_table.join(pred_prob_table, lambda label, prob: (label if label > 0 else 0, prob))
            predict_result = predict_result.join(pred_label_table, lambda x, y: (x[0], x[1], y))
        else:
            predict_result = actual_label_table.join(pred_label_table, lambda a_label, p_label: (a_label, None, p_label))
        return predict_result

    def load_model(self, model_table_name, model_namespace):
        LOGGER.info("@ load host model from name/ns" + ", " + str(model_table_name) + ", " + str(model_namespace))
        model_parameters = load_model_parameters(model_table_name, model_namespace)
        self.host_model.restore_model(model_parameters)

    def save_model(self, model_table_name, model_namespace):
        LOGGER.info("@ save host model to name/ns" + ", " + str(model_table_name) + ", " + str(model_namespace))
        _ = save_model_parameters(self.host_model.get_model_parameters(), model_table_name, model_namespace)


class HeteroPlainFTLHost(HeteroFTLHost):

    def __init__(self, host: PlainFTLHostModel, model_param: FTLModelParam, transfer_variable: HeteroFTLTransferVariable):
        super(HeteroPlainFTLHost, self).__init__(host, model_param, transfer_variable)

    def fit(self, host_data):
        LOGGER.info("@ start host fit")

        host_x, overlap_indexes = self.prepare_data(host_data)

        LOGGER.debug("host_x： " + str(host_x.shape))
        LOGGER.debug("overlap_indexes: " + str(len(overlap_indexes)))

        self.host_model.set_batch(host_x, overlap_indexes)
        while self.n_iter_ < self.max_iter:
            host_comp = self.host_model.send_components()
            self._do_remote(host_comp, name=self.transfer_variable.host_component_list.name,
                            tag=self.transfer_variable.generate_transferid(self.transfer_variable.host_component_list, self.n_iter_),
                            role=consts.GUEST,
                            idx=-1)

            guest_comp = self._do_get(name=self.transfer_variable.guest_component_list.name,
                                      tag=self.transfer_variable.generate_transferid(self.transfer_variable.guest_component_list, self.n_iter_),
                                      idx=-1)[0]

            self.host_model.receive_components(guest_comp)

            is_stop = self._do_get(name=self.transfer_variable.is_stopped.name,
                                   tag=self.transfer_variable.generate_transferid(self.transfer_variable.is_stopped, self.n_iter_),
                                   idx=-1)[0]

            LOGGER.info("@ time: " + str(time.time()) + ", ep: " + str(self.n_iter_) + ", converged: " + str(is_stop))
            self.n_iter_ += 1
            if is_stop:
                break


"""
Centralized encryption scheme with an arbiter in the loop for decryption.
"""


class HeteroEncryptFTLHost(HeteroFTLHost):

    def __init__(self, host, model_param: FTLModelParam, transfer_variable: HeteroFTLTransferVariable):
        super(HeteroEncryptFTLHost, self).__init__(host, model_param, transfer_variable)
        self.host_model: EncryptedFTLHostModel = host

    def _precompute(self):
        pass

    def fit(self, host_data):
        LOGGER.info("@ start host fit")
        # get public key from arbiter
        public_key = self._do_get(name=self.transfer_variable.paillier_pubkey.name,
                                  tag=self.transfer_variable.generate_transferid(self.transfer_variable.paillier_pubkey),
                                  idx=-1)[0]

        host_x, overlap_indexes = self.prepare_data(host_data)

        LOGGER.debug("host_x： " + str(host_x.shape))
        LOGGER.debug("overlap_indexes: " + str(len(overlap_indexes)))

        self.host_model.set_batch(host_x, overlap_indexes)
        self.host_model.set_public_key(public_key)

        start_time = time.time()
        while self.n_iter_ < self.max_iter:
            host_comp = self.host_model.send_components()
            self._do_remote(host_comp, name=self.transfer_variable.host_component_list.name,
                            tag=self.transfer_variable.generate_transferid(self.transfer_variable.host_component_list, self.n_iter_),
                            role=consts.GUEST,
                            idx=-1)

            guest_comp = self._do_get(name=self.transfer_variable.guest_component_list.name,
                                      tag=self.transfer_variable.generate_transferid(
                                          self.transfer_variable.guest_component_list, self.n_iter_),
                                      idx=-1)[0]
            self.host_model.receive_components(guest_comp)

            self._precompute()

            encrypt_host_gradients = self.host_model.send_gradients()
            self._do_remote(encrypt_host_gradients, name=self.transfer_variable.encrypt_host_gradient.name,
                            tag=self.transfer_variable.generate_transferid(self.transfer_variable.encrypt_host_gradient, self.n_iter_),
                            role=consts.ARBITER,
                            idx=-1)

            decrypt_host_gradients = self._do_get(name=self.transfer_variable.decrypt_host_gradient.name,
                                                 tag=self.transfer_variable.generate_transferid(
                                                     self.transfer_variable.decrypt_host_gradient, self.n_iter_),
                                                 idx=-1)[0]
            self.host_model.receive_gradients(decrypt_host_gradients)

            is_stop = self._do_get(name=self.transfer_variable.is_encrypted_ftl_stopped.name,
                                   tag=self.transfer_variable.generate_transferid(self.transfer_variable.is_encrypted_ftl_stopped, self.n_iter_),
                                   idx=-1)[0]

            LOGGER.info("@ time: " + str(time.time()) + ", ep: " + str(self.n_iter_) + ", converged: " + str(is_stop))
            self.n_iter_ += 1
            if is_stop:
                break

        end_time = time.time()
        LOGGER.info("@ running time: " + str(end_time - start_time))


class FasterHeteroEncryptFTLHost(HeteroEncryptFTLHost):

    def __init__(self, host, model_param: FTLModelParam, transfer_variable: HeteroFTLTransferVariable):
        super(FasterHeteroEncryptFTLHost, self).__init__(host, model_param, transfer_variable)
        self.host_model: FasterEncryptedFTLHostModel = host

    def _precompute(self):
        LOGGER.info("@ start host precompute")

        host_precomputed_comp = self.host_model.send_precomputed_components()
        self._do_remote(host_precomputed_comp, name=self.transfer_variable.host_precomputed_comp_list.name,
                        tag=self.transfer_variable.generate_transferid(self.transfer_variable.host_precomputed_comp_list,
                                                                       self.n_iter_),
                        role=consts.GUEST,
                        idx=-1)

        guest_precomputed_comp = self._do_get(name=self.transfer_variable.guest_precomputed_comp_list.name,
                                              tag=self.transfer_variable.generate_transferid(
                                                  self.transfer_variable.guest_precomputed_comp_list, self.n_iter_),
                                              idx=-1)[0]
        self.host_model.receive_precomputed_components(guest_precomputed_comp)


"""
Decentralized encryption scheme without arbiter in the loop.
"""


class HeteroDecentralizedEncryptFTLHost(HeteroFTLHost):

    def __init__(self, host, model_param: FTLModelParam, transfer_variable: HeteroFTLTransferVariable):
        super(HeteroDecentralizedEncryptFTLHost, self).__init__(host, model_param, transfer_variable)
        self.host_model: EncryptedFTLHostModel = host
        self.public_key = None
        self.private_key = None
        self.guest_public_key = None

    def _precompute(self):
        pass

    def prepare_encryption_key_pair(self):
        LOGGER.info("@ start host prepare encryption key pair")

        self.public_key, self.private_key = generate_encryption_key_pair()
        # exchange public_key with guest
        self._do_remote(self.public_key, name=self.transfer_variable.host_public_key.name,
                        tag=self.transfer_variable.generate_transferid(self.transfer_variable.host_public_key,
                                                                       self.n_iter_),
                        role=consts.GUEST,
                        idx=-1)

        self.guest_public_key = self._do_get(name=self.transfer_variable.guest_public_key.name,
                                  tag=self.transfer_variable.generate_transferid(
                                      self.transfer_variable.guest_public_key, self.n_iter_),
                                  idx=-1)[0]

    def fit(self, host_data):
        LOGGER.info("@ start host fit")
        self.prepare_encryption_key_pair()
        host_x, overlap_indexes = self.prepare_data(host_data)

        LOGGER.debug("host_x： " + str(host_x.shape))
        LOGGER.debug("overlap_indexes: " + str(len(overlap_indexes)))

        self.host_model.set_batch(host_x, overlap_indexes)
        self.host_model.set_public_key(self.public_key)
        self.host_model.set_guest_public_key(self.guest_public_key)
        self.host_model.set_private_key(self.private_key)

        start_time = time.time()
        while self.n_iter_ < self.max_iter:

            # Stage 1: compute and encrypt components (using host public key) required by guest to
            #          calculate gradients and loss.
            LOGGER.debug("@ Stage 1: ")
            host_comp = self.host_model.send_components()
            LOGGER.debug("send enc host_comp: " + create_shape_msg(host_comp))
            self._do_remote(host_comp, name=self.transfer_variable.host_component_list.name,
                            tag=self.transfer_variable.generate_transferid(self.transfer_variable.host_component_list, self.n_iter_),
                            role=consts.GUEST,
                            idx=-1)

            # Stage 2: receive guest components in encrypted form (encrypted by guest public key),
            #          and calculate host gradients in encrypted form (encrypted by guest public key),
            #          and send them to guest for decryption
            LOGGER.debug("@ Stage 2: ")
            guest_comp = self._do_get(name=self.transfer_variable.guest_component_list.name,
                                      tag=self.transfer_variable.generate_transferid(
                                          self.transfer_variable.guest_component_list, self.n_iter_),
                                      idx=-1)[0]
            LOGGER.debug("receive enc guest_comp: " + create_shape_msg(guest_comp))
            self.host_model.receive_components(guest_comp)

            self._precompute()

            # calculate host gradients in encrypted form (encrypted by guest public key)
            encrypt_host_gradients = self.host_model.send_gradients()
            LOGGER.debug("send encrypt_guest_gradients: " + create_shape_msg(encrypt_host_gradients))

            # add random mask to encrypt_host_gradients and send them to guest for decryption
            masked_enc_host_gradients, gradients_masks = add_random_mask_for_list_of_values(encrypt_host_gradients)

            LOGGER.debug("send masked_enc_host_gradients: " + create_shape_msg(masked_enc_host_gradients))
            self._do_remote(masked_enc_host_gradients, name=self.transfer_variable.masked_enc_host_gradients.name,
                            tag=self.transfer_variable.generate_transferid(self.transfer_variable.masked_enc_host_gradients, self.n_iter_),
                            role=consts.GUEST,
                            idx=-1)

            # Stage 3: receive and then decrypt masked encrypted guest gradients and masked encrypted guest loss,
            #          and send them to guest
            LOGGER.debug("@ Stage 3: ")
            masked_enc_guest_gradients = self._do_get(name=self.transfer_variable.masked_enc_guest_gradients.name,
                                                   tag=self.transfer_variable.generate_transferid(self.transfer_variable.masked_enc_guest_gradients, self.n_iter_),
                                                   idx=-1)[0]

            masked_enc_guest_loss = self._do_get(name=self.transfer_variable.masked_enc_loss.name,
                                                   tag=self.transfer_variable.generate_transferid(self.transfer_variable.masked_enc_loss, self.n_iter_),
                                                   idx=-1)[0]

            masked_dec_guest_gradients = self.__decrypt_gradients(masked_enc_guest_gradients)
            masked_dec_guest_loss = self.__decrypt_loss(masked_enc_guest_loss)

            LOGGER.debug("send masked_dec_guest_gradients: " + create_shape_msg(masked_dec_guest_gradients))
            self._do_remote(masked_dec_guest_gradients, name=self.transfer_variable.masked_dec_guest_gradients.name,
                            tag=self.transfer_variable.generate_transferid(self.transfer_variable.masked_dec_guest_gradients, self.n_iter_),
                            role=consts.GUEST,
                            idx=-1)
            LOGGER.debug("send masked_dec_guest_loss: " + str(masked_dec_guest_loss))
            self._do_remote(masked_dec_guest_loss, name=self.transfer_variable.masked_dec_loss.name,
                            tag=self.transfer_variable.generate_transferid(self.transfer_variable.masked_dec_loss, self.n_iter_),
                            role=consts.GUEST,
                            idx=-1)

            # Stage 4: receive masked but decrypted host gradients from guest and remove mask,
            #          and update host model parameters using these gradients.
            LOGGER.debug("@ Stage 4: ")
            masked_dec_host_gradients = self._do_get(name=self.transfer_variable.masked_dec_host_gradients.name,
                                                     tag=self.transfer_variable.generate_transferid(
                                                         self.transfer_variable.masked_dec_host_gradients, self.n_iter_),
                                                     idx=-1)[0]
            LOGGER.debug("receive masked_dec_host_gradients: " + create_shape_msg(masked_dec_host_gradients))

            cleared_dec_host_gradients = remove_random_mask_from_list_of_values(masked_dec_host_gradients, gradients_masks)

            # update host model parameters using these gradients.
            self.host_model.receive_gradients(cleared_dec_host_gradients)

            # Stage 5: determine whether training is terminated.
            LOGGER.debug("@ Stage 5: ")
            is_stop = self._do_get(name=self.transfer_variable.is_decentralized_enc_ftl_stopped.name,
                                   tag=self.transfer_variable.generate_transferid(self.transfer_variable.is_decentralized_enc_ftl_stopped, self.n_iter_),
                                   idx=-1)[0]

            LOGGER.info("@ time: " + str(time.time()) + ", ep: " + str(self.n_iter_) + ", converged: " + str(is_stop))
            self.n_iter_ += 1
            if is_stop:
                break

        end_time = time.time()
        LOGGER.info("@ running time: " + str(end_time - start_time))

    def __decrypt_gradients(self, encrypt_gradients):
        return distribute_decrypt_matrix(self.private_key, encrypt_gradients[0]), decrypt_array(self.private_key, encrypt_gradients[1])

    def __decrypt_loss(self, encrypt_loss):
        return decrypt_scalar(self.private_key, encrypt_loss)


class FasterHeteroDecentralizedEncryptFTLHost(HeteroDecentralizedEncryptFTLHost):

    def __init__(self, host, model_param: FTLModelParam, transfer_variable: HeteroFTLTransferVariable):
        super(FasterHeteroDecentralizedEncryptFTLHost, self).__init__(host, model_param, transfer_variable)
        self.host_model: FasterEncryptedFTLHostModel = host

    def _precompute(self):
        LOGGER.debug("@ start precompute")

        host_precomputed_comp = self.host_model.send_precomputed_components()
        self._do_remote(host_precomputed_comp, name=self.transfer_variable.host_precomputed_comp_list.name,
                        tag=self.transfer_variable.generate_transferid(self.transfer_variable.host_precomputed_comp_list,
                                                                       self.n_iter_),
                        role=consts.GUEST,
                        idx=-1)

        guest_precomputed_comp = self._do_get(name=self.transfer_variable.guest_precomputed_comp_list.name,
                                              tag=self.transfer_variable.generate_transferid(
                                                  self.transfer_variable.guest_precomputed_comp_list, self.n_iter_),
                                              idx=-1)[0]
        self.host_model.receive_precomputed_components(guest_precomputed_comp)


class HostFactory(object):

    @classmethod
    def create(cls, ftl_model_param: FTLModelParam, transfer_variable: HeteroFTLTransferVariable, ftl_local_model):
        if ftl_model_param.is_encrypt:
            if ftl_model_param.enc_ftl == "dct_enc_ftl":
                # decentralized encrypted ftl host
                LOGGER.debug("@ create decentralized encrypted ftl_host")
                host_model = EncryptedFTLHostModel(local_model=ftl_local_model, model_param=ftl_model_param)
                host = HeteroDecentralizedEncryptFTLHost(host_model, ftl_model_param, transfer_variable)
            elif ftl_model_param.enc_ftl == "dct_enc_ftl2":
                # decentralized encrypted faster ftl host
                LOGGER.debug("@ create decentralized encrypted faster ftl_host")
                host_model = FasterEncryptedFTLHostModel(local_model=ftl_local_model, model_param=ftl_model_param)
                host = FasterHeteroDecentralizedEncryptFTLHost(host_model, ftl_model_param, transfer_variable)
            elif ftl_model_param.enc_ftl == "enc_ftl2":
                # encrypted faster ftl host
                LOGGER.debug("@ create encrypted faster ftl_host")
                host_model = FasterEncryptedFTLHostModel(local_model=ftl_local_model, model_param=ftl_model_param)
                host = FasterHeteroEncryptFTLHost(host_model, ftl_model_param, transfer_variable)
            else:
                # encrypted ftl host
                LOGGER.debug("@ create encrypted ftl_host")
                host_model = EncryptedFTLHostModel(local_model=ftl_local_model, model_param=ftl_model_param)
                host = HeteroEncryptFTLHost(host_model, ftl_model_param, transfer_variable)

        else:
            # plain ftl host
            LOGGER.debug("@ create plain ftl_host")
            host_model = PlainFTLHostModel(local_model=ftl_local_model, model_param=ftl_model_param)
            host = HeteroPlainFTLHost(host_model, ftl_model_param, transfer_variable)
        return host


