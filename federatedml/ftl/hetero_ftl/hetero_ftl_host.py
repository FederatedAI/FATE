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
import tensorflow as tf

from arch.api.proto import ftl_model_meta_pb2, ftl_model_param_pb2
from arch.api.utils import log_utils
from federatedml.ftl.data_util.common_data_util import overlapping_samples_converter, create_table, \
    convert_instance_table_to_dict, convert_instance_table_to_array, \
    add_random_mask_for_list_of_values, remove_random_mask_from_list_of_values
from federatedml.ftl.data_util.log_util import create_shape_msg
from federatedml.ftl.eggroll_computation.helper import distribute_decrypt_matrix
from federatedml.ftl.encrypted_ftl import EncryptedFTLHostModel
from federatedml.ftl.encryption.encryption import generate_encryption_key_pair, decrypt_scalar, decrypt_array
from federatedml.ftl.faster_encrypted_ftl import FasterEncryptedFTLHostModel
from federatedml.ftl.hetero_ftl.hetero_ftl_base import HeteroFTLParty
from federatedml.ftl.plain_ftl import PlainFTLHostModel
from federatedml.param.ftl_param import FTLParam, FTLModelParam
from federatedml.util import consts
from federatedml.util.transfer_variable.hetero_ftl_transfer_variable import HeteroFTLTransferVariable

LOGGER = log_utils.getLogger()


class HeteroFTLHost(HeteroFTLParty):

    def __init__(self):
        super(HeteroFTLHost, self).__init__()

        self.model_param = FTLParam()
        self.transfer_variable = HeteroFTLTransferVariable()
        self.n_iter_ = 0
        self.model_param_name = 'FTLModelParam'
        self.model_meta_name = 'FTLModelMeta'

    def _init_host_model(self, params):
        raise NotImplementedError("method init must be define")

    def _init_model(self, params):
        self.max_iter = params.max_iter
        self.local_model = self._create_local_model(params)
        self.host_model = self._init_host_model(params)
        self.predict_param = params.predict_param

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

    def _fit(self, host_x, overlap_indexes):
        raise NotImplementedError("method init must be define")

    def fit(self, host_data):
        LOGGER.info("@ start host fit")

        host_x, overlap_indexes = self.prepare_data(host_data)

        LOGGER.debug("host_x： " + str(host_x.shape))
        LOGGER.debug("overlap_indexes: " + str(len(overlap_indexes)))

        start_time = time.time()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            self.local_model.set_session(sess)
            sess.run(init)

            self._fit(host_x, overlap_indexes)

            self.model_parameters = self.local_model.get_model_parameters()
        end_time = time.time()
        LOGGER.info("@ running time: " + str(end_time - start_time))

    def classified(self, prob_table, threshold):
        """
        convert a probability table into a predicted class table.
        """
        predict_table = prob_table.mapValues(lambda x: 1 if x > threshold else 0)
        return predict_table

    def predict(self, host_data):
        LOGGER.info("@ start host predict")
        features, labels, instances_indexes = convert_instance_table_to_array(host_data)
        host_x = np.squeeze(features)
        LOGGER.debug("host_x： " + str(host_x.shape))

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            self.local_model.set_session(sess)
            sess.run(init)

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
        pred_label_table = self.classified(pred_prob_table, self.predict_param.threshold)
        if self.predict_param.with_proba:
            predict_result = actual_label_table.join(pred_prob_table,
                                                     lambda label, prob: (label, prob))
            predict_result = predict_result.join(pred_label_table, lambda x, y: [x[0], x[1], y])
        else:
            predict_result = actual_label_table.join(pred_label_table,
                                                     lambda a_label, p_label: [a_label, None, p_label])
        return predict_result

    def export_model(self):
        hyperparameters = self.model_parameters['hyperparameters']

        meta_obj = ftl_model_meta_pb2.FTLModelMeta(input_dim=hyperparameters['input_dim'],
                                                   encode_dim=hyperparameters['hidden_dim'],
                                                   learning_rate=hyperparameters['learning_rate'])
        param_obj = ftl_model_param_pb2.FTLModelParam(weight_hidden=self.model_parameters['Wh'],
                                                      bias_hidden=self.model_parameters['bh'],
                                                      weight_output=self.model_parameters['Wo'],
                                                      bias_output=self.model_parameters['bo'])
        result = {
            self.model_meta_name: meta_obj,
            self.model_param_name: param_obj
        }
        return result

    def _load_model(self, model_dict):
        model = list(model_dict.get('model').values())[0]
        meta_obj = model.get(self.model_meta_name)
        param_obj = model.get(self.model_param_name)
        self.model_parameters = {
            'Wh': param_obj.weight_hidden,
            'bh': param_obj.bias_hidden,
            'Wo': param_obj.weight_output,
            'bo': param_obj.bias_output,
            'hyperparameters': {
                'input_dim': meta_obj.input_dim,
                'hidden_dim': meta_obj.encode_dim,
                'learning_rate': meta_obj.learning_rate
            }
        }
        tf.reset_default_graph()
        self.local_model.restore_model(self.model_parameters)


class HeteroPlainFTLHost(HeteroFTLHost):

    def __init__(self):
        super(HeteroPlainFTLHost, self).__init__()

    def _init_host_model(self, params):
        return PlainFTLHostModel(local_model=self.local_model, model_param=params)

    def _fit(self, host_x, overlap_indexes):
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

    def __init__(self):
        super(HeteroEncryptFTLHost, self).__init__()

    def _init_host_model(self, params):
        return EncryptedFTLHostModel(local_model=self.local_model, model_param=params)

    def _precompute(self):
        pass

    def _fit(self, host_x, overlap_indexes):
        # get public key from arbiter
        public_key = self._do_get(name=self.transfer_variable.paillier_pubkey.name,
                                  tag=self.transfer_variable.generate_transferid(self.transfer_variable.paillier_pubkey),
                                  idx=-1)[0]

        self.host_model.set_batch(host_x, overlap_indexes)
        self.host_model.set_public_key(public_key)

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


class FasterHeteroEncryptFTLHost(HeteroEncryptFTLHost):

    def __init__(self):
        super(FasterHeteroEncryptFTLHost, self).__init__()

    def _init_host_model(self, params):
        return FasterEncryptedFTLHostModel(local_model=self.local_model, model_param=params)

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

    def __init__(self):
        super(HeteroDecentralizedEncryptFTLHost, self).__init__()
        self.public_key = None
        self.private_key = None
        self.guest_public_key = None

    def _init_host_model(self, params):
        return EncryptedFTLHostModel(local_model=self.local_model, model_param=params)

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

    def _fit(self, host_x, overlap_indexes):
        self.prepare_encryption_key_pair()

        self.host_model.set_batch(host_x, overlap_indexes)
        self.host_model.set_public_key(self.public_key)
        self.host_model.set_guest_public_key(self.guest_public_key)
        self.host_model.set_private_key(self.private_key)

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

    def __decrypt_gradients(self, encrypt_gradients):
        return distribute_decrypt_matrix(self.private_key, encrypt_gradients[0]), decrypt_array(self.private_key, encrypt_gradients[1])

    def __decrypt_loss(self, encrypt_loss):
        return decrypt_scalar(self.private_key, encrypt_loss)


class FasterHeteroDecentralizedEncryptFTLHost(HeteroDecentralizedEncryptFTLHost):

    def __init__(self):
        super(FasterHeteroDecentralizedEncryptFTLHost, self).__init__()

    def _init_host_model(self, params):
        return FasterEncryptedFTLHostModel(local_model=self.local_model, model_param=params)

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
    def create(cls, ftl_param: FTLParam, ftl_model_param: FTLModelParam):
        if ftl_model_param.is_encrypt:
            if ftl_model_param.enc_ftl == "dct_enc_ftl":
                # decentralized encrypted ftl host
                LOGGER.debug("@ create decentralized encrypted ftl_host")
                host = HeteroDecentralizedEncryptFTLHost()
            elif ftl_model_param.enc_ftl == "dct_enc_ftl2":
                # decentralized encrypted faster ftl host
                LOGGER.debug("@ create decentralized encrypted faster ftl_host")
                host = FasterHeteroDecentralizedEncryptFTLHost()
            elif ftl_model_param.enc_ftl == "enc_ftl2":
                # encrypted faster ftl host
                LOGGER.debug("@ create encrypted faster ftl_host")
                host = FasterHeteroEncryptFTLHost()
            else:
                # encrypted ftl host
                LOGGER.debug("@ create encrypted ftl_host")
                host = HeteroEncryptFTLHost()

        else:
            # plain ftl host
            LOGGER.debug("@ create plain ftl_host")
            host = HeteroPlainFTLHost()
        host._init_model(ftl_param)
        return host


