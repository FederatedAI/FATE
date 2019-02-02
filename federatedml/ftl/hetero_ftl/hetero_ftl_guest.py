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

import numpy as np
import time
from federatedml.ftl.plain_ftl import PlainFTLGuestModel
from federatedml.ftl.encrypted_ftl import EncryptedFTLGuestModel
from federatedml.ftl.faster_encrypted_ftl import FasterEncryptedFTLGuestModel
from federatedml.ftl.data_util.common_data_util import overlapping_samples_converter, load_model_parameters, \
    save_model_parameters, convert_instance_table_to_dict, convert_instance_table_to_array
from federatedml.ftl.hetero_ftl.hetero_ftl_base import HeteroFTLParty
from federatedml.util import consts
from federatedml.util.transfer_variable import HeteroFTLTransferVariable
from federatedml.optim.convergence import DiffConverge
from federatedml.param.param import FTLModelParam
from arch.api.utils import log_utils
LOGGER = log_utils.getLogger()


class HeteroFTLGuest(HeteroFTLParty):

    def __init__(self, guest: PlainFTLGuestModel, model_param: FTLModelParam, transfer_variable: HeteroFTLTransferVariable):
        super(HeteroFTLGuest, self).__init__()
        self.guest_model = guest
        self.model_param = model_param
        self.transfer_variable = transfer_variable
        self.max_iter = model_param.max_iter
        self.n_iter_ = 0
        self.converge_func = DiffConverge(eps=model_param.eps)

    def set_converge_function(self, converge_func):
        self.converge_func = converge_func

    def prepare_data(self, guest_data):
        LOGGER.info("@ start guest prepare_data")
        guest_features_dict, guest_label_dict, guest_sample_indexes = convert_instance_table_to_dict(guest_data)
        guest_sample_indexes = np.array(guest_sample_indexes)
        LOGGER.debug("@ send guest_sample_indexes shape" + str(guest_sample_indexes.shape))
        self._do_remote(guest_sample_indexes,
                        name=self.transfer_variable.guest_sample_indexes.name,
                        tag=self.transfer_variable.generate_transferid(self.transfer_variable.guest_sample_indexes),
                        role=consts.HOST,
                        idx=-1)
        host_sample_indexes = self._do_get(name=self.transfer_variable.host_sample_indexes.name,
                                           tag=self.transfer_variable.generate_transferid(self.transfer_variable.host_sample_indexes),
                                           idx=-1)[0]

        LOGGER.debug("@ receive host_sample_indexes len" + str(len(host_sample_indexes)))
        guest_features, overlap_indexes, non_overlap_indexes, guest_label = overlapping_samples_converter(
            guest_features_dict, guest_sample_indexes, host_sample_indexes, guest_label_dict)
        return guest_features, overlap_indexes, non_overlap_indexes, guest_label

    def predict(self, guest_data):
        LOGGER.info("@ start guest predict")
        features, labels, instances_indexes = convert_instance_table_to_array(guest_data)
        guest_x = np.squeeze(features)
        guest_y = np.expand_dims(labels, axis=1)
        LOGGER.debug("guest_x, guest_y: " + str(guest_x.shape) + ", " + str(guest_y.shape))

        host_prob = self._do_get(name=self.transfer_variable.host_prob.name,
                                 tag=self.transfer_variable.generate_transferid(self.transfer_variable.host_prob),
                                 idx=-1)[0]

        self.guest_model.set_batch(guest_x, guest_y)
        pred_prob = self.guest_model.predict(host_prob)
        LOGGER.debug("pred_prob: " + str(pred_prob.shape))

        self._do_remote(pred_prob,
                        name=self.transfer_variable.pred_prob.name,
                        tag=self.transfer_variable.generate_transferid(self.transfer_variable.pred_prob),
                        role=consts.HOST,
                        idx=-1)
        return None

    def load_model(self, model_table_name, model_namespace):
        LOGGER.info("@ load guest model from name/ns" + ", " + str(model_table_name) + ", " + str(model_namespace))
        model_parameters = load_model_parameters(model_table_name, model_namespace)
        self.guest_model.restore_model(model_parameters)

    def save_model(self, model_table_name, model_namespace):
        LOGGER.info("@ save guest model to name/ns" + ", " + str(model_table_name) + ", " + str(model_namespace))
        _ = save_model_parameters(self.guest_model.get_model_parameters(), model_table_name, model_namespace)


class HeteroPlainFTLGuest(HeteroFTLGuest):

    def __init__(self, guest: PlainFTLGuestModel, model_param: FTLModelParam, transfer_variable: HeteroFTLTransferVariable):
        super(HeteroPlainFTLGuest, self).__init__(guest, model_param, transfer_variable)

    def fit(self, guest_data):
        LOGGER.info("@ start guest fit")

        guest_x, overlap_indexes, non_overlap_indexes, guest_y = self.prepare_data(guest_data)

        LOGGER.debug("guest_x： " + str(guest_x.shape))
        LOGGER.debug("guest_y： " + str(guest_y.shape))
        LOGGER.debug("overlap_indexes: " + str(len(overlap_indexes)))
        LOGGER.debug("non_overlap_indexes: " + str(len(non_overlap_indexes)))

        self.guest_model.set_batch(guest_x, guest_y, non_overlap_indexes, overlap_indexes)
        is_stop = False
        while self.n_iter_ < self.max_iter:
            guest_comp = self.guest_model.send_components()
            LOGGER.debug("send guest_comp: " + str(guest_comp[0].shape) + ", " + str(guest_comp[1].shape) + ", " + str(
                guest_comp[2].shape))
            self._do_remote(guest_comp, name=self.transfer_variable.guest_component_list.name,
                            tag=self.transfer_variable.generate_transferid(self.transfer_variable.guest_component_list, self.n_iter_),
                            role=consts.HOST,
                            idx=-1)

            host_comp = self._do_get(name=self.transfer_variable.host_component_list.name,
                                     tag=self.transfer_variable.generate_transferid(self.transfer_variable.host_component_list, self.n_iter_),
                                     idx=-1)[0]
            LOGGER.debug("receive host_comp: " + str(host_comp[0].shape) + ", " + str(host_comp[1].shape) + ", " + str(host_comp[2].shape))
            self.guest_model.receive_components(host_comp)

            loss = self.guest_model.send_loss()
            if self.converge_func.is_converge(loss):
                is_stop = True

            self._do_remote(is_stop, name=self.transfer_variable.is_stopped.name,
                            tag=self.transfer_variable.generate_transferid(self.transfer_variable.is_stopped, self.n_iter_),
                            role=consts.HOST,
                            idx=-1)
            LOGGER.info("@ time: " + str(time.time()) + ", ep:" + str(self.n_iter_) + ", loss:" + str(loss))
            LOGGER.info("@ converged: " + str(is_stop))
            self.n_iter_ += 1
            if is_stop:
                break


class HeteroEncryptFTLGuest(HeteroFTLGuest):

    def __init__(self, guest, model_param: FTLModelParam, transfer_variable: HeteroFTLTransferVariable):
        super(HeteroEncryptFTLGuest, self).__init__(guest, model_param, transfer_variable)
        self.guest_model: EncryptedFTLGuestModel = guest

    def _precompute(self):
        pass

    def fit(self, guest_data):
        LOGGER.info("@ start guest fit")
        public_key = self._do_get(name=self.transfer_variable.paillier_pubkey.name,
                                  tag=self.transfer_variable.generate_transferid(self.transfer_variable.paillier_pubkey),
                                  idx=-1)[0]

        guest_x, overlap_indexes, non_overlap_indexes, guest_y = self.prepare_data(guest_data)

        LOGGER.debug("guest_x： " + str(guest_x.shape))
        LOGGER.debug("guest_y： " + str(guest_y.shape))
        LOGGER.debug("overlap_indexes: " + str(len(overlap_indexes)))
        LOGGER.debug("non_overlap_indexes: " + str(len(non_overlap_indexes)))

        self.guest_model.set_batch(guest_x, guest_y, non_overlap_indexes, overlap_indexes)
        self.guest_model.set_public_key(public_key)

        start_time = time.time()
        while self.n_iter_ < self.max_iter:
            guest_comp = self.guest_model.send_components()

            LOGGER.debug("send guest_comp: " + str(guest_comp[0].shape) + ", " + str(guest_comp[1].shape) + ", " + str(
                guest_comp[2].shape))
            self._do_remote(guest_comp, name=self.transfer_variable.guest_component_list.name,
                            tag=self.transfer_variable.generate_transferid(self.transfer_variable.guest_component_list, self.n_iter_),
                            role=consts.HOST,
                            idx=-1)

            host_comp = self._do_get(name=self.transfer_variable.host_component_list.name,
                                     tag=self.transfer_variable.generate_transferid(self.transfer_variable.host_component_list, self.n_iter_),
                                     idx=-1)[0]

            LOGGER.debug("receive host_comp: " + str(host_comp[0].shape) + ", " + str(host_comp[1].shape) + ", " + str(host_comp[2].shape))
            self.guest_model.receive_components(host_comp)

            self._precompute()

            encrypt_guest_gradients = self.guest_model.send_gradients()
            self._do_remote(encrypt_guest_gradients, name=self.transfer_variable.encrypt_guest_gradient.name,
                            tag=self.transfer_variable.generate_transferid(self.transfer_variable.encrypt_guest_gradient, self.n_iter_),
                            role=consts.ARBITER,
                            idx=-1)

            decrypt_guest_gradient = self._do_get(name=self.transfer_variable.decrypt_guest_gradient.name,
                                                  tag=self.transfer_variable.generate_transferid(self.transfer_variable.decrypt_guest_gradient, self.n_iter_),
                                                  idx=-1)[0]
            self.guest_model.receive_gradients(decrypt_guest_gradient)

            encrypt_loss = self.guest_model.send_loss()
            self._do_remote(encrypt_loss, name=self.transfer_variable.encrypt_loss.name,
                            tag=self.transfer_variable.generate_transferid(self.transfer_variable.encrypt_loss, self.n_iter_),
                            role=consts.ARBITER,
                            idx=-1)

            is_stop = self._do_get(name=self.transfer_variable.is_encrypted_ftl_stopped.name,
                                   tag=self.transfer_variable.generate_transferid(self.transfer_variable.is_encrypted_ftl_stopped, self.n_iter_),
                                   idx=-1)[0]

            LOGGER.info("@ time: " + str(time.time()) + ", ep: " + str(self.n_iter_) + ", converged：" + str(is_stop))
            self.n_iter_ += 1
            if is_stop:
                break

        end_time = time.time()
        LOGGER.info("@ running time: " + str(end_time - start_time))


class FasterHeteroEncryptFTLGuest(HeteroEncryptFTLGuest):

    def __init__(self, guest, model_param: FTLModelParam, transfer_variable: HeteroFTLTransferVariable):
        super(FasterHeteroEncryptFTLGuest, self).__init__(guest, model_param, transfer_variable)
        self.guest_model: FasterEncryptedFTLGuestModel = guest

    def _precompute(self):

        guest_precomputed_comp = self.guest_model.send_precomputed_components()
        self._do_remote(guest_precomputed_comp, name=self.transfer_variable.guest_precomputed_comp_list.name,
                        tag=self.transfer_variable.generate_transferid(self.transfer_variable.guest_precomputed_comp_list,
                                                                       self.n_iter_),
                        role=consts.HOST,
                        idx=-1)

        host_precomputed_comp = self._do_get(name=self.transfer_variable.host_precomputed_comp_list.name,
                                 tag=self.transfer_variable.generate_transferid(
                                     self.transfer_variable.host_precomputed_comp_list, self.n_iter_),
                                 idx=-1)[0]
        self.guest_model.receive_precomputed_components(host_precomputed_comp)


