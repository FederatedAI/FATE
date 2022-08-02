#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

import pickle
import numpy as np
from federatedml.nn.hetero_nn.backend.tf_keras.interactive.drop_out import DropOut
from federatedml.util.fixpoint_solver import FixedPointEncoder
from federatedml.nn.backend.tf_keras.nn_model import build_keras
from federatedml.secureprotol.paillier_tensor import PaillierTensor
from federatedml.nn.hetero_nn.backend.tf_keras.interactive.dense_model import GuestDenseModel
from federatedml.nn.hetero_nn.backend.tf_keras.interactive.dense_model import HostDenseModel
from federatedml.nn.hetero_nn.util import random_number_generator
from federatedml.protobuf.generated.hetero_nn_model_param_pb2 import InteractiveLayerParam
from federatedml.secureprotol import PaillierEncrypt
from federatedml.util import consts, LOGGER


class InterActiveGuestDenseLayer(object):

    def __init__(self, params=None, config_type=None, layer_config=None, host_num=1):

        self.host_num = host_num
        self.nn_define = layer_config
        self.layer_config = layer_config
        self.config_type = config_type
        self.host_input_shapes = []
        self.guest_input_shape = None
        self.model = None
        self.rng_generator = random_number_generator.RandomNumberGenerator()
        self.model_builder = build_keras
        self.transfer_variable = None
        self.learning_rate = params.interactive_layer_lr
        self.encrypted_host_dense_output = None
        self.encrypted_host_input = None
        self.guest_input = None
        self.guest_output = None
        self.host_output = None
        self.dense_output_data = None
        self.guest_model = None
        self.host_model_list = []
        self.batch_size = None
        self.partitions = 0
        self.do_backward_select_strategy = False
        self.encrypted_host_input_cached = None
        self.drop_out_keep_rate = params.drop_out_keep_rate
        self.drop_out = None

        self.fixed_point_encoder = None if params.floating_point_precision is None else FixedPointEncoder(
            2 ** params.floating_point_precision)

        self.sync_output_unit = False

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable

    def set_backward_select_strategy(self):
        self.do_backward_select_strategy = True

    def set_batch(self, batch_size):
        self.batch_size = batch_size

    def set_partition(self, partition):
        self.partitions = partition

    def __build_model(self, restore_stage=False):

        use_torch = self.config_type == consts.pytorch_backend
        if use_torch:
            LOGGER.debug('interaction layer now is using pytorch configuration')

        # for multi host cases
        for i in range(self.host_num):
            host_model = HostDenseModel()
            host_model.build(self.host_input_shapes[i], self.layer_config, self.model_builder, restore_stage, use_torch)
            host_model.set_learning_rate(self.learning_rate)
            self.host_model_list.append(host_model)

        LOGGER.debug('host model list is {}'.format(self.host_model_list))

        self.guest_model = GuestDenseModel()
        self.guest_model.build(self.guest_input_shape, self.layer_config, self.model_builder, restore_stage, use_torch)
        self.guest_model.set_learning_rate(self.learning_rate)

        if use_torch:
            # torch gradient computation is different from keras
            self.guest_model.disable_mean_gradient()
            for host_model in self.host_model_list:
                host_model.disable_mean_gradient()

        if self.do_backward_select_strategy:
            self.guest_model.set_backward_selective_strategy()
            self.guest_model.set_batch(self.batch_size)
            for host_model in self.host_model_list:
                host_model.set_backward_selective_strategy()
                host_model.set_batch(self.batch_size)

    def forward(self, guest_input, epoch=0, batch=0, train=True):

        LOGGER.info("interactive layer start forward propagation of epoch {} batch {}".format(epoch, batch))

        host_inputs = self.get_host_encrypted_forward_from_host(epoch, batch, idx=-1)

        host_bottom_inputs_tensor = []
        host_input_shapes = []
        for i in host_inputs:
            pt = PaillierTensor(i)
            host_bottom_inputs_tensor.append(pt)
            host_input_shapes.append(pt.shape[1])
            LOGGER.debug('encrypted host input shape is {}'.format(pt.shape))

        self.host_input_shapes = host_input_shapes

        if self.guest_model is None:
            LOGGER.info("building interactive layers' training model")
            self.guest_input_shape = guest_input.shape[1] if guest_input is not None else 0
            self.__build_model()
        if not self.partitions:
            self.partitions = host_bottom_inputs_tensor[0].partitions

        if not self.sync_output_unit:
            self.sync_output_unit = True
            for idx in range(self.host_num):
                self.sync_interactive_layer_output_unit(self.host_model_list[idx].output_shape[0], idx=idx)

        host_output = self.forward_interactive(host_bottom_inputs_tensor, epoch, batch, train)
        guest_output = self.guest_model.forward_dense(guest_input)

        if not self.guest_model.empty:
            dense_output_data = host_output + PaillierTensor(guest_output, partitions=self.partitions)
        else:
            dense_output_data = host_output

        self.dense_output_data = dense_output_data
        self.guest_output = guest_output
        self.host_output = host_output

        LOGGER.info("start to get interactive layer's activation output of epoch {} batch {}".format(epoch, batch))
        # all model has same activation layer
        activation_out = self.host_model_list[0].forward_activation(self.dense_output_data.numpy())
        for host_model in self.host_model_list[1:]:
            host_model.activation_input = self.dense_output_data.numpy()
        LOGGER.info("end to get interactive layer's activation output of epoch {} batch {}".format(epoch, batch))

        if train and self.drop_out:
            activation_out = self.drop_out.forward(activation_out)

        return activation_out

    def backward(self, output_gradient, selective_ids, epoch, batch):

        if selective_ids:
            for host_model in self.host_model_list:
                host_model.select_backward_sample(selective_ids)
            self.guest_model.select_backward_sample(selective_ids)
            if self.drop_out:
                self.drop_out.select_backward_sample(selective_ids)

        if self.do_backward_select_strategy:
            # send to all host
            self.sync_backward_select_info(selective_ids, len(output_gradient), epoch, batch, -1)

        if len(output_gradient) > 0:
            LOGGER.debug("interactive layer start backward propagation of epoch {} batch {}".format(epoch, batch))
            activation_backward = self.host_model_list[0].backward_activation()[0]
            activation_gradient = output_gradient * activation_backward
            if self.drop_out:
                activation_gradient = self.drop_out.backward(activation_gradient)

            LOGGER.debug("interactive layer update guest weight of epoch {} batch {}".format(epoch, batch))
            # update guest model
            guest_input_gradient = self.update_guest(activation_gradient)

            LOGGER.debug('update host model weights')
            for idx, host_model in enumerate(self.host_model_list):
                # update host models
                host_weight_gradient, acc_noise = self.backward_interactive(host_model, activation_gradient, epoch,
                                                                            batch,
                                                                            host_idx=idx)
                host_input_gradient = self.update_host(host_model, activation_gradient, host_weight_gradient, acc_noise)
                self.send_host_backward_to_host(host_input_gradient.get_obj(), epoch, batch, idx=idx)

            return guest_input_gradient
        else:
            return []

    def _create_drop_out(self, shape):
        if self.drop_out_keep_rate and self.drop_out_keep_rate != 1:
            if not self.drop_out:
                self.drop_out = DropOut(noise_shape=shape, rate=self.drop_out_keep_rate)
                self.drop_out.set_partition(self.partitions)
                if self.do_backward_select_strategy:
                    self.drop_out.do_backward_select_strategy()

            self.drop_out.generate_mask()

    def sync_interactive_layer_output_unit(self, shape, idx=0):
        self.transfer_variable.interactive_layer_output_unit.remote(shape,
                                                                    role=consts.HOST,
                                                                    idx=idx)

    def sync_backward_select_info(self, selective_ids, gradient_len, epoch, batch, idx):
        self.transfer_variable.selective_info.remote((selective_ids, gradient_len),
                                                     role=consts.HOST,
                                                     idx=idx,
                                                     suffix=(epoch, batch,))

    def send_host_backward_to_host(self, host_error, epoch, batch, idx):
        self.transfer_variable.host_backward.remote(host_error,
                                                    role=consts.HOST,
                                                    idx=idx,
                                                    suffix=(epoch, batch,))

    def update_guest(self, activation_gradient):
        input_gradient = self.guest_model.get_input_gradient(activation_gradient)
        weight_gradient = self.guest_model.get_weight_gradient(activation_gradient)
        self.guest_model.apply_update(weight_gradient)

        return input_gradient

    def update_host(self, host_model, activation_gradient, weight_gradient, acc_noise):
        activation_gradient_tensor = PaillierTensor(activation_gradient, partitions=self.partitions)
        input_gradient = host_model.get_input_gradient(activation_gradient_tensor, acc_noise,
                                                       encoder=self.fixed_point_encoder)
        host_model.update_weight(weight_gradient)
        host_model.update_bias(activation_gradient)

        return input_gradient

    def forward_interactive(self, encrypted_host_input, epoch, batch, train=True):

        LOGGER.info("get encrypted dense output of host model of epoch {} batch {}".format(epoch, batch))
        mask_table_list = []
        guest_nosies = []

        host_idx = 0
        for model, host_bottom_input in zip(self.host_model_list, encrypted_host_input):

            encrypted_fw = model.forward_dense(host_bottom_input, self.fixed_point_encoder)
            mask_table = None
            if train:
                self._create_drop_out(encrypted_fw.shape)
                if self.drop_out:
                    mask_table = self.drop_out.generate_mask_table()
                if mask_table:
                    encrypted_fw = encrypted_fw.select_columns(mask_table)
                    mask_table_list.append(mask_table)

            guest_forward_noise = self.rng_generator.fast_generate_random_number(encrypted_fw.shape,
                                                                                 encrypted_fw.partitions,
                                                                                 keep_table=mask_table)

            if self.fixed_point_encoder:
                encrypted_fw += guest_forward_noise.encode(self.fixed_point_encoder)
            else:
                encrypted_fw += guest_forward_noise

            guest_nosies.append(guest_forward_noise)

            self.send_guest_encrypted_forward_output_with_noise_to_host(encrypted_fw.get_obj(), epoch, batch,
                                                                        idx=host_idx)
            if mask_table:
                self.send_interactive_layer_drop_out_table(mask_table, epoch, batch, idx=host_idx)

            host_idx += 1

        LOGGER.info("get decrypted dense output of host model of epoch {} batch {}".format(epoch, batch))

        # get list from hosts
        decrypted_dense_outputs = self.get_guest_decrypted_forward_from_host(epoch, batch, idx=-1)
        merge_output = None
        for idx, (outputs, noise) in enumerate(zip(decrypted_dense_outputs, guest_nosies)):
            out = PaillierTensor(outputs) - noise
            # handle mask table
            if len(mask_table_list) != 0:
                out = PaillierTensor(out.get_obj().join(mask_table_list[idx], self.expand_columns))
            if merge_output is None:
                merge_output = out
            else:
                merge_output = merge_output + out

        return merge_output

        # if mask_table:
        #     out = PaillierTensor(decrypted_dense_output) - guest_forward_noise
        #     out = out.get_obj().join(mask_table, self.expand_columns)
        #     return PaillierTensor(out)
        # else:
        #     return PaillierTensor(decrypted_dense_output) - guest_forward_noise

    def backward_interactive(self, host_model, activation_gradient, epoch, batch, host_idx):
        LOGGER.info("get encrypted weight gradient of epoch {} batch {}".format(epoch, batch))
        encrypted_weight_gradient = host_model.get_weight_gradient(activation_gradient,
                                                                   encoder=self.fixed_point_encoder)
        if self.fixed_point_encoder:
            encrypted_weight_gradient = self.fixed_point_encoder.decode(encrypted_weight_gradient)

        noise_w = self.rng_generator.generate_random_number(encrypted_weight_gradient.shape)
        self.transfer_variable.encrypted_guest_weight_gradient.remote(encrypted_weight_gradient + noise_w,
                                                                      role=consts.HOST,
                                                                      idx=host_idx,
                                                                      suffix=(epoch, batch,))

        LOGGER.info("get decrypted weight graident of epoch {} batch {}".format(epoch, batch))
        decrypted_weight_gradient = self.transfer_variable.decrypted_guest_weight_gradient.get(idx=host_idx,
                                                                                               suffix=(epoch, batch,))

        decrypted_weight_gradient -= noise_w

        encrypted_acc_noise = self.get_encrypted_acc_noise_from_host(epoch, batch, idx=host_idx)

        return decrypted_weight_gradient, encrypted_acc_noise

    def get_host_encrypted_forward_from_host(self, epoch, batch, idx=0):
        return self.transfer_variable.encrypted_host_forward.get(idx=idx,
                                                                 suffix=(epoch, batch,))

    def send_guest_encrypted_forward_output_with_noise_to_host(self, encrypted_guest_forward_with_noise, epoch, batch,
                                                               idx):
        return self.transfer_variable.encrypted_guest_forward.remote(encrypted_guest_forward_with_noise,
                                                                     role=consts.HOST,
                                                                     idx=idx,
                                                                     suffix=(epoch, batch,))

    def send_interactive_layer_drop_out_table(self, mask_table, epoch, batch, idx):
        return self.transfer_variable.drop_out_table.remote(mask_table,
                                                            role=consts.HOST,
                                                            idx=idx,
                                                            suffix=(epoch, batch,))

    def get_guest_decrypted_forward_from_host(self, epoch, batch, idx=0):
        return self.transfer_variable.decrypted_guest_fowrad.get(idx=idx,
                                                                 suffix=(epoch, batch,))

    def get_encrypted_acc_noise_from_host(self, epoch, batch, idx=0):
        return self.transfer_variable.encrypted_acc_noise.get(idx=idx,
                                                              suffix=(epoch, batch,))

    @staticmethod
    def expand_columns(tensor, keep_array):
        shape = keep_array.shape
        tensor = np.reshape(tensor, (tensor.size,))
        keep = np.reshape(keep_array, (keep_array.size,))
        ret_tensor = []
        idx = 0
        for x in keep:
            if x == 0:
                ret_tensor.append(0)
            else:
                ret_tensor.append(tensor[idx])
                idx += 1

        return np.reshape(np.array(ret_tensor), shape)

    def export_model(self):
        interactive_layer_param = InteractiveLayerParam()
        interactive_layer_param.interactive_guest_saved_model_bytes = self.guest_model.export_model()
        host_model_bytes = []
        for m in self.host_model_list:
            host_model_bytes.append(m.export_model())
        interactive_layer_param.interactive_host_saved_model_bytes.extend(host_model_bytes)
        interactive_layer_param.host_input_shape.extend(self.host_input_shapes)
        interactive_layer_param.guest_input_shape = self.guest_input_shape

        return interactive_layer_param

    def restore_model(self, interactive_layer_param):

        self.host_input_shapes = list(interactive_layer_param.host_input_shape)
        self.guest_input_shape = interactive_layer_param.guest_input_shape
        self.__build_model(restore_stage=True)
        self.guest_model.restore_model(interactive_layer_param.interactive_guest_saved_model_bytes)
        for idx, m in enumerate(self.host_model_list):
            m.restore_model(interactive_layer_param.interactive_host_saved_model_bytes[idx])  # restore from bytes


class InteractiveHostDenseLayer(object):
    def __init__(self, params):
        self.acc_noise = None
        self.learning_rate = params.interactive_layer_lr
        self.encrypter = self.generate_encrypter(params)
        self.transfer_variable = None
        self.partitions = 1
        self.input_shape = None
        self.output_unit = None
        self.rng_generator = random_number_generator.RandomNumberGenerator()
        self.do_backward_select_strategy = False
        self.drop_out_keep_rate = params.drop_out_keep_rate

        self.fixed_point_encoder = None if params.floating_point_precision is None else FixedPointEncoder(
            2 ** params.floating_point_precision)
        self.mask_table = None

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable

    def set_partition(self, partition):
        self.partitions = partition

    def set_backward_select_strategy(self):
        self.do_backward_select_strategy = True

    def forward(self, host_input, epoch=0, batch=0, train=True):

        LOGGER.info("forward propagation: encrypt host_bottom_output of epoch {} batch {}".format(epoch, batch))
        host_input = PaillierTensor(host_input, partitions=self.partitions)

        encrypted_host_input = host_input.encrypt(self.encrypter)
        self.send_host_encrypted_forward_to_guest(encrypted_host_input.get_obj(), epoch, batch)

        encrypted_guest_forward = PaillierTensor(self.get_guest_encrypted_forwrad_from_guest(epoch, batch))

        decrypted_guest_forward = encrypted_guest_forward.decrypt(self.encrypter)
        if self.fixed_point_encoder:
            decrypted_guest_forward = decrypted_guest_forward.decode(self.fixed_point_encoder)

        if self.input_shape is None:
            self.input_shape = host_input.shape[1]
            self.output_unit = self.get_interactive_layer_output_unit()

        if self.acc_noise is None:
            # self.input_shape = host_input.shape[1]
            # self.output_unit = self.get_interactive_layer_output_unit()
            self.acc_noise = np.zeros((self.input_shape, self.output_unit))

        mask_table = None
        if train and self.drop_out_keep_rate and self.drop_out_keep_rate < 1:
            mask_table = self.get_interactive_layer_drop_out_table(epoch, batch)

        if mask_table:
            decrypted_guest_forward_with_noise = decrypted_guest_forward + \
                (host_input * self.acc_noise).select_columns(mask_table)
            self.mask_table = mask_table
        else:
            decrypted_guest_forward_with_noise = decrypted_guest_forward + (host_input * self.acc_noise)

        self.send_decrypted_guest_forward_with_noise_to_guest(decrypted_guest_forward_with_noise.get_obj(), epoch,
                                                              batch)

    def backward(self, epoch, batch):
        do_backward = True
        selective_ids = []
        if self.do_backward_select_strategy:
            selective_ids, do_backward = self.sync_backward_select_info(epoch, batch)

        if not do_backward:
            return [], selective_ids

        encrypted_guest_weight_gradient = self.get_guest_encrypted_weight_gradient_from_guest(epoch, batch)

        LOGGER.info("decrypt weight gradient of epoch {} batch {}".format(epoch, batch))
        decrypted_guest_weight_gradient = self.encrypter.recursive_decrypt(encrypted_guest_weight_gradient)

        noise_weight_gradient = self.rng_generator.generate_random_number((self.input_shape, self.output_unit))

        decrypted_guest_weight_gradient += noise_weight_gradient / self.learning_rate

        self.send_guest_decrypted_weight_gradient_to_guest(decrypted_guest_weight_gradient, epoch, batch)

        LOGGER.info("encrypt acc_noise of epoch {} batch {}".format(epoch, batch))
        encrypted_acc_noise = self.encrypter.recursive_encrypt(self.acc_noise)
        self.send_encrypted_acc_noise_to_guest(encrypted_acc_noise, epoch, batch)

        self.acc_noise += noise_weight_gradient
        host_input_gradient = PaillierTensor(self.get_host_backward_from_guest(epoch, batch))

        host_input_gradient = host_input_gradient.decrypt(self.encrypter)

        if self.fixed_point_encoder:
            host_input_gradient = host_input_gradient.decode(self.fixed_point_encoder).numpy()
        else:
            host_input_gradient = host_input_gradient.numpy()

        return host_input_gradient, selective_ids

    def sync_backward_select_info(self, epoch, batch):
        selective_ids, do_backward = self.transfer_variable.selective_info.get(idx=0,
                                                                               suffix=(epoch, batch,))

        return selective_ids, do_backward

    def send_encrypted_acc_noise_to_guest(self, encrypted_acc_noise, epoch, batch):
        self.transfer_variable.encrypted_acc_noise.remote(encrypted_acc_noise,
                                                          idx=0,
                                                          role=consts.GUEST,
                                                          suffix=(epoch, batch,))

    def get_interactive_layer_output_unit(self):
        return self.transfer_variable.interactive_layer_output_unit.get(idx=0)

    def get_guest_encrypted_weight_gradient_from_guest(self, epoch, batch):
        encrypted_guest_weight_gradient = self.transfer_variable.encrypted_guest_weight_gradient.get(
            idx=0, suffix=(epoch, batch,))

        return encrypted_guest_weight_gradient

    def get_interactive_layer_drop_out_table(self, epoch, batch):
        return self.transfer_variable.drop_out_table.get(idx=0,
                                                         suffix=(epoch, batch,))

    def send_host_encrypted_forward_to_guest(self, encrypted_host_input, epoch, batch):
        self.transfer_variable.encrypted_host_forward.remote(encrypted_host_input,
                                                             idx=0,
                                                             role=consts.GUEST,
                                                             suffix=(epoch, batch,))

    def send_guest_decrypted_weight_gradient_to_guest(self, decrypted_guest_weight_gradient, epoch, batch):
        self.transfer_variable.decrypted_guest_weight_gradient.remote(decrypted_guest_weight_gradient,
                                                                      idx=0,
                                                                      role=consts.GUEST,
                                                                      suffix=(epoch, batch,))

    def get_host_backward_from_guest(self, epoch, batch):
        host_backward = self.transfer_variable.host_backward.get(idx=0,
                                                                 suffix=(epoch, batch,))

        return host_backward

    def get_guest_encrypted_forwrad_from_guest(self, epoch, batch):
        encrypted_guest_forward = self.transfer_variable.encrypted_guest_forward.get(idx=0,
                                                                                     suffix=(epoch, batch,))

        return encrypted_guest_forward

    def send_decrypted_guest_forward_with_noise_to_guest(self, decrypted_guest_forward_with_noise, epoch, batch):
        self.transfer_variable.decrypted_guest_fowrad.remote(decrypted_guest_forward_with_noise,
                                                             idx=0,
                                                             role=consts.GUEST,
                                                             suffix=(epoch, batch,))

    def generate_encrypter(self, param):
        LOGGER.info("generate encrypter")
        if param.encrypt_param.method.lower() == consts.PAILLIER.lower():
            encrypter = PaillierEncrypt()
            encrypter.generate_key(param.encrypt_param.key_length)
        else:
            raise NotImplementedError("encrypt method not supported yet!!!")

        return encrypter

    def export_model(self):
        interactive_layer_param = InteractiveLayerParam()
        interactive_layer_param.acc_noise = pickle.dumps(self.acc_noise)

        return interactive_layer_param

    def restore_model(self, interactive_layer_param):
        self.acc_noise = pickle.loads(interactive_layer_param.acc_noise)
