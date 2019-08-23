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

from __future__ import print_function

from arch.api.utils import log_utils
from federatedml.homo import MeanAggregator
from federatedml.homo import Model
from federatedml.homo import SynchronizedPartyWeights
from federatedml.homo import ModelAggregate, GradientAggregate
from federatedml.homo import ModelBroadcast, GradientBroadcast
from federatedml.param.homo_param import HomoParam
from federatedml.util.transfer_variable.homo_transfer_variable import HomeModelTransferVariable
from federatedml.util.consts import PAILLIER, RANDOM_PADS, NONE
from federatedml.model_base import ModelBase
from federatedml.util import consts
from federatedml.homo import RandomPadding

LOGGER = log_utils.getLogger()


class RandomPadsModel(ModelBase):

    def __init__(self, model: Model):
        super().__init__()
        self.model = model

        self.transfer_variable: HomeModelTransferVariable = HomeModelTransferVariable()

        self._iter_index = 0

        self.flowid = None

    def fit(self, data_instance):
        pass

    def save_model(self):
        self.model.save_model()

    def load_model(self):
        self.model.load_model()

    def _init_model(self, model):
        pass

    def predict(self, data_inst):
        pass


class ArbiterRandomPadsModel(RandomPadsModel):

    def __init__(self, model, params: HomoParam):
        super().__init__(model)
        self.aggregator = MeanAggregator

        self.party_weights = []  # The first one is guest weight, host weights for otherwise
        self.max_iter = params.max_iter

        # params for encryption
        self.encrypt_param = params.encrypt_param
        self.encrypt_method = self.encrypt_param.method

        self.weights = None




    def fit(self, data_instances):

        if self.use_encrypt:
            RandomPadding(self.transfer_variable).cipher_create_arbiter()

        LOGGER.info("synchronizing party weights")
        self.party_weights = SynchronizedPartyWeights.from_transfer_variable(transfer_variable=self.transfer_variable) \
            .arbiter_fn()
        LOGGER.info(f"synchronized party weights: {self.party_weights}")

        model_agg_fn = ModelAggregate.from_transfer_variable(transfer_variable=self.transfer_variable).arbiter_get_weights
        model_bc_fn = ModelBroadcast.from_transfer_variable(transfer_variable=self.transfer_variable).arbiter_get_weights
        gradient_agg_fn = \
            GradientAggregate.from_transfer_variable(transfer_variable=self.transfer_variable).arbiter_get_weights
        gradient_bc_fn = GradientBroadcast.from_transfer_variable(transfer_variable=self.transfer_variable).arbiter_get_weights

        for iter_num in range(self.max_iter):

            for batch_num in range(self.num_batch):

                if self.fit_mode == consts.MODEL_AGG:

                    mean_weights = model_agg_fn(party_weights=self.party_weights,
                                                tag_suffix=f"epoch_{self._iter_index}")
                self.weights = mean_weights
                model_bc_fn(model_weights=self.weights,
                            paillier_ciphers=paillier_ciphers,
                            tag_suffix=f"epoch_{self._iter_index}")
            else:
                mean_weights = gradient_agg_fn(party_weights=self.party_weights,
                                               paillier_ciphers=paillier_ciphers,
                                               tag_suffix=f"epoch_{self._iter_index}")
                # apply gradient to arbiter model
                self.model.apply_gradient_weights(mean_weights)

                # get arbiter model for transfer
                self.weights = self.model.get_model_weights()
                gradient_bc_fn(model_weights=self.weights,
                               paillier_ciphers=paillier_ciphers,
                               tag_suffix=f"epoch_{self._iter_index}")
            self._iter_index += 1



class GuestModel(HomoModel):

    def __init__(self, model, params: HomoParam):
        super().__init__(model)

        self.party_weight = params.party_weight
        self.max_iter = params.max_iter
        self.model_agg_iter = params.model_agg_iter
        self.model.set_additional_params(batch_size=params.batch_size,
                                         num_batch=params.num_batch)

        # params for encryption
        self.encrypt_params = params.encrypt_param
        self.encrypt_method = self.encrypt_params.method

        self._prepare()

    def _prepare(self):

        LOGGER.info(f"use encrypt method: {self.encrypt_method}")
        if self.encrypt_method == RANDOM_PADS:
            self.pads_cipher = RandomPadding(self.transfer_variable)
            self.pads_cipher.cipher_create_guest()

        elif self.encrypt_method == PAILLIER:
            pass  # do nothing

        elif self.encrypt_method == NONE:
            pass

        else:
            raise NotImplementedError(f"encrypt method {self.encrypt_method} not implemented")

        LOGGER.info(f"synchronizing party weights, local: {self.party_weight}")
        self.party_weights_norm = \
            SynchronizedPartyWeights.from_transfer_variable(transfer_variable=self.transfer_variable)\
            .send_party_weight(self.party_weight)
        LOGGER.info(f"synchronized party weights: {self.party_weights_norm}")

    def fit(self, data_instances):
        print(self.max_iter)
        max_iter = self.max_iter
        model_agg_fn = ModelAggregate.from_transfer_variable(transfer_variable=self.transfer_variable).send_party_weight
        model_bc_fn = ModelBroadcast.from_transfer_variable(transfer_variable=self.transfer_variable).send_party_weight
        gradient_agg_fn = GradientAggregate.from_transfer_variable(transfer_variable=self.transfer_variable).send_party_weight
        gradient_bc_fn = GradientBroadcast.from_transfer_variable(transfer_variable=self.transfer_variable).send_party_weight
        while self._iter_index < max_iter:

            if self.fit_mode == consts.MODEL_AGG:
                # train local model
                self.model.train_local(data_instances)

                # get weighs to transfer
                transfer_weights = self.model.get_model_weights()
            else:
                # get batch data
                batch = data_instances.train.next_batch(128)

                # get weighs to transfer(gradient)
                transfer_weights = self.model.get_gradient_weights(batch)
                train_accuracy = self.model.get_batch_accuracy(batch)
                print("training accuracy %g" % train_accuracy)

            # encrypt before transfer to arbiter
            if self.encrypt_method == RANDOM_PADS:
                self.pads_cipher.encrypt(transfer_weights)


            if self.fit_mode == consts.MODEL_AGG:
                model_agg_fn(weights=transfer_weights, tag_suffix=f"epoch_{self._iter_index}")
                remote_wgt = model_bc_fn(tag_suffix=f"epoch_{self._iter_index}")
            else:
                gradient_agg_fn(weights=transfer_weights, tag_suffix=f"epoch_{self._iter_index}")
                remote_wgt = gradient_bc_fn(tag_suffix=f"epoch_{self._iter_index}")

            self.model.assign_model_weights(remote_wgt)

            self._iter_index += 1

    def save_model(self):
        self.model.save_model()


class HostModel(HomoModel):

    def __init__(self, model, params: HomoParam):
        super().__init__(model)

        self.party_weight = params.party_weight
        self.max_iter = params.max_iter
        self.model.set_additional_params(batch_size=params.batch_size,
                                         num_batch=params.num_batch)

        # params for encryption
        self.encrypt_param = params.encrypt_param
        self.encrypt_method = params.encrypt_param.method

        if self.encrypt_method == PAILLIER:
            self.host_use_paillier_encrypt = True
        self._prepare()

    def _prepare(self):

        LOGGER.info(f"use encrypt method: {self.encrypt_method}")
        if self.encrypt_method == RANDOM_PADS:
            self.pads_cipher = RandomPadding(self.transfer_variable)
            self.pads_cipher.cipher_create_host()

        elif self.encrypt_method == PAILLIER:

            if self.host_use_paillier_encrypt:
                self.encrypt_operator.set_public_key(public_key=pubkey)

        elif self.encrypt_method == NONE:
            pass

        else:
            raise NotImplementedError(f"encrypt method {self.encrypt_method} not implemented")

        LOGGER.info(f"synchronizing party weights, local: {self.party_weight}")
        self.party_weights_norm = \
            SynchronizedPartyWeights.from_transfer_variable(transfer_variable=self.transfer_variable) \
            .host_call(self.party_weight)
        LOGGER.info(f"synchronized party weights: {self.party_weights_norm}")

    def fit(self, data_instances):
        max_iter = self.max_iter
        model_agg_fn = ModelAggregate.from_transfer_variable(transfer_variable=self.transfer_variable).host_call
        model_bc_fn = ModelBroadcast.from_transfer_variable(transfer_variable=self.transfer_variable).host_call
        gradient_agg_fn = GradientAggregate.from_transfer_variable(transfer_variable=self.transfer_variable).host_call
        gradient_bc_fn = GradientBroadcast.from_transfer_variable(transfer_variable=self.transfer_variable).host_call
        while self._iter_index < max_iter:

            if self.fit_mode == consts.MODEL_AGG:
                # train local model
                self.model.train_local(data_instances)

                # get weighs to transfer
                transfer_weights = self.model.get_model_weights()
            else:
                # get batch data
                batch = data_instances.train.next_batch(128)

                # get weighs to transfer(gradient)
                transfer_weights = self.model.get_gradient_weights(batch)
                train_accuracy = self.model.get_batch_accuracy(batch)
                print("training accuracy %g" % train_accuracy)

            # encrypt before transfer to arbiter
            if self.encrypt_method == RANDOM_PADS:
                # Pads cipher encrypt model before sending to arbiter
                self.pads_cipher.encrypt(transfer_weights)

            if self.fit_mode == consts.MODEL_AGG:
                model_agg_fn(weights=transfer_weights, tag_suffix=f"epoch_{self._iter_index}")
                remote_wgt = model_bc_fn(tag_suffix=f"epoch_{self._iter_index}")
            else:
                gradient_agg_fn(weights=transfer_weights, tag_suffix=f"epoch_{self._iter_index}")
                remote_wgt = gradient_bc_fn(tag_suffix=f"epoch_{self._iter_index}")

            self.model.assign_model_weights(remote_wgt)

            self._iter_index += 1
