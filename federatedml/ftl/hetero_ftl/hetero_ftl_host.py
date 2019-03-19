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
from federatedml.ftl.encrypted_ftl import EncryptedFTLHostModel
from federatedml.ftl.plain_ftl import PlainFTLHostModel
from federatedml.ftl.common.data_util import overlapping_samples_converter, load_model_parameters, \
    save_model_parameters, create_table, convert_instance_table_to_dict, convert_instance_table_to_array
from federatedml.ftl.hetero_ftl.hetero_ftl_base import HeteroFTLParty
from federatedml.util import consts
from federatedml.util.transfer_variable import HeteroFTLTransferVariable
from federatedml.evaluation import Evaluation
from federatedml.param.param import FTLModelParam
from arch.api.utils import log_utils
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

    def evaluate(self, labels, pred_prob, pred_labels, evaluate_param):
        LOGGER.info("@ start host evaluate")
        predict_res = None
        if evaluate_param.classi_type == consts.BINARY:
            predict_res = pred_prob
        elif evaluate_param.classi_type == consts.MULTY:
            predict_res = pred_labels
        else:
            LOGGER.warning("unknown classification type, return None as evaluation results")

        eva = Evaluation(evaluate_param.classi_type)
        eva_report = eva.report(labels, predict_res, evaluate_param.metrics, evaluate_param.thresholds,
                          evaluate_param.pos_label)

        LOGGER.info("@ evaluation report:" + str(eva_report))
        return eva_report

    def predict(self, host_data, predict_param):
        LOGGER.info("@ start host predict")
        features, labels, instances_indexes = convert_instance_table_to_array(host_data)
        host_x = np.squeeze(features)
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


class HeteroEncryptFTLHost(HeteroFTLHost):

    def __init__(self, host: EncryptedFTLHostModel, model_param: FTLModelParam, transfer_variable: HeteroFTLTransferVariable):
        super(HeteroEncryptFTLHost, self).__init__(host, model_param, transfer_variable)
        self.host_model = host

    def fit(self, host_data):
        LOGGER.info("@ start host fit")
        # get public key from arbiter
        public_key = self._do_get(name=self.transfer_variable.paillier_pubkey.name,
                                  tag=self.transfer_variable.generate_transferid(self.transfer_variable.paillier_pubkey),
                                  idx=-1)[0]

        host_x, overlap_indexes = self.prepare_data(host_data)

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

            encrypt_host_gradients = self.host_model.send_gradients()
            self._do_remote(encrypt_host_gradients, name=self.transfer_variable.encrypt_host_gradient.name,
                            tag=self.transfer_variable.generate_transferid(self.transfer_variable.encrypt_host_gradient, self.n_iter_),
                            role=consts.ARBITER,
                            idx=-1)

            decrypt_host_gradient = self._do_get(name=self.transfer_variable.decrypt_host_gradient.name,
                                                 tag=self.transfer_variable.generate_transferid(
                                                     self.transfer_variable.decrypt_host_gradient, self.n_iter_),
                                                 idx=-1)[0]
            self.host_model.receive_gradients(decrypt_host_gradient)

            is_stop = self._do_get(name=self.transfer_variable.is_encrypted_ftl_stopped.name,
                                   tag=self.transfer_variable.generate_transferid(self.transfer_variable.is_encrypted_ftl_stopped, self.n_iter_),
                                   idx=-1)[0]

            LOGGER.info("@ time: " + str(time.time()) + ", ep: " + str(self.n_iter_) + ", converged: " + str(is_stop))
            self.n_iter_ += 1
            if is_stop:
                break

