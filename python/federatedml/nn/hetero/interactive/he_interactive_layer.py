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
import torch
from torch import autograd
from federatedml.nn.hetero.interactive.base import InteractiveLayerGuest, InteractiveLayerHost
from federatedml.nn.hetero.nn_component.torch_model import backward_loss
from federatedml.nn.backend.torch.interactive import InteractiveLayer
from federatedml.nn.backend.torch.serialization import recover_sequential_from_dict
from federatedml.util.fixpoint_solver import FixedPointEncoder
from federatedml.protobuf.generated.hetero_nn_model_param_pb2 import InteractiveLayerParam
from federatedml.secureprotol import PaillierEncrypt
from federatedml.util import consts, LOGGER
from federatedml.nn.hetero.interactive.utils.numpy_layer import NumpyDenseLayerGuest, NumpyDenseLayerHost
from federatedml.secureprotol.paillier_tensor import PaillierTensor
from federatedml.nn.hetero.nn_component.torch_model import TorchNNModel
from federatedml.transfer_variable.base_transfer_variable import BaseTransferVariables
from fate_arch.session import computing_session as session
from federatedml.nn.backend.utils.rng import RandomNumberGenerator


PLAINTEXT = False


class HEInteractiveTransferVariable(BaseTransferVariables):
    def __init__(self, flowid=0):
        super().__init__(flowid)
        self.decrypted_guest_forward = self._create_variable(
            name='decrypted_guest_forward', src=['host'], dst=['guest'])
        self.decrypted_guest_weight_gradient = self._create_variable(
            name='decrypted_guest_weight_gradient', src=['host'], dst=['guest'])
        self.encrypted_acc_noise = self._create_variable(
            name='encrypted_acc_noise', src=['host'], dst=['guest'])
        self.encrypted_guest_forward = self._create_variable(
            name='encrypted_guest_forward', src=['guest'], dst=['host'])
        self.encrypted_guest_weight_gradient = self._create_variable(
            name='encrypted_guest_weight_gradient', src=['guest'], dst=['host'])
        self.encrypted_host_forward = self._create_variable(
            name='encrypted_host_forward', src=['host'], dst=['guest'])
        self.host_backward = self._create_variable(
            name='host_backward', src=['guest'], dst=['host'])
        self.selective_info = self._create_variable(
            name="selective_info", src=["guest"], dst=["host"])
        self.drop_out_info = self._create_variable(
            name="drop_out_info", src=["guest"], dst=["host"])
        self.drop_out_table = self._create_variable(
            name="drop_out_table", src=["guest"], dst=["host"])
        self.interactive_layer_output_unit = self._create_variable(
            name="interactive_layer_output_unit", src=["guest"], dst=["host"])


class DropOut(object):
    def __init__(self, rate, noise_shape):
        self._keep_rate = rate
        self._noise_shape = noise_shape
        self._batch_size = noise_shape[0]
        self._mask = None
        self._partition = None
        self._mask_table = None
        self._select_mask_table = None
        self._do_backward_select = False

        self._mask_table_cache = {}

    def forward(self, X):
        if X.shape == self._mask.shape:
            forward_x = X * self._mask / self._keep_rate
        else:
            forward_x = X * self._mask[0: len(X)] / self._keep_rate
        return forward_x

    def backward(self, grad):

        if self._do_backward_select:
            self._mask = self._select_mask_table[0: grad.shape[0]]
            self._select_mask_table = self._select_mask_table[grad.shape[0]:]
            return grad * self._mask / self._keep_rate
        else:
            if grad.shape == self._mask.shape:
                return grad * self._mask / self._keep_rate
            else:
                return grad * self._mask[0: grad.shape[0]] / self._keep_rate

    def generate_mask(self):
        self._mask = np.random.uniform(
            low=0, high=1, size=self._noise_shape) < self._keep_rate

    def generate_mask_table(self, shape):
        # generate mask table according to samples shape, because in some
        # batches, sample_num < batch_size
        if shape == self._noise_shape:
            _mask_table = session.parallelize(
                self._mask, include_key=False, partition=self._partition)
        else:
            _mask_table = session.parallelize(
                self._mask[0: shape[0]], include_key=False, partition=self._partition)

        return _mask_table

    def set_partition(self, partition):
        self._partition = partition

    def select_backward_sample(self, select_ids):
        select_mask_table = self._mask[np.array(select_ids)]
        if self._select_mask_table is not None:
            self._select_mask_table = np.vstack(
                (self._select_mask_table, select_mask_table))
        else:
            self._select_mask_table = select_mask_table

    def do_backward_select_strategy(self):
        self._do_backward_select = True


class HEInteractiveLayerGuest(InteractiveLayerGuest):

    def __init__(self, params=None, layer_config=None, host_num=1):
        super(HEInteractiveLayerGuest, self).__init__(params)

        # transfer var
        self.host_num = host_num
        self.layer_config = layer_config
        self.transfer_variable = HEInteractiveTransferVariable()
        self.plaintext = PLAINTEXT
        self.layer_config = layer_config
        self.host_input_shapes = []
        self.rng_generator = RandomNumberGenerator()
        self.learning_rate = params.interactive_layer_lr

        # cached tensor
        self.guest_tensor = None
        self.host_tensors = None
        self.dense_output_data_require_grad = None
        self.activation_out_require_grad = None

        # model
        self.model: InteractiveLayer = None
        self.guest_model = None
        self.host_model_list = []
        self.batch_size = None
        self.partitions = 0
        self.do_backward_select_strategy = False
        self.optimizer = None

        # drop out
        self.drop_out_initiated = False
        self.drop_out = None
        self.drop_out_keep_rate = None

        self.fixed_point_encoder = None if params.floating_point_precision is None else FixedPointEncoder(
            2 ** params.floating_point_precision)

        self.send_output_unit = False

        # float64
        self.float64 = False

    """
    Init functions
    """

    def set_flow_id(self, flow_id):
        self.transfer_variable.set_flowid(flow_id)

    def set_backward_select_strategy(self):
        self.do_backward_select_strategy = True

    def set_batch(self, batch_size):
        self.batch_size = batch_size

    def set_partition(self, partition):
        self.partitions = partition

    def _build_model(self):

        if self.model is None:
            raise ValueError('torch interactive model is not initialized!')

        for i in range(self.host_num):
            host_model = NumpyDenseLayerHost()
            host_model.build(self.model.host_model[i])
            host_model.set_learning_rate(self.learning_rate)
            self.host_model_list.append(host_model)

        self.guest_model = NumpyDenseLayerGuest()
        self.guest_model.build(self.model.guest_model)
        self.guest_model.set_learning_rate(self.learning_rate)

        if self.do_backward_select_strategy:
            self.guest_model.set_backward_selective_strategy()
            self.guest_model.set_batch(self.batch_size)
            for host_model in self.host_model_list:
                host_model.set_backward_selective_strategy()
                host_model.set_batch(self.batch_size)

    """
    Drop out functions
    """

    def init_drop_out_parameter(self):
        if isinstance(self.model.param_dict['dropout'], float):
            self.drop_out_keep_rate = 1 - self.model.param_dict['dropout']
        else:
            self.drop_out_keep_rate = -1
        self.transfer_variable.drop_out_info.remote(
            self.drop_out_keep_rate, idx=-1, suffix=('dropout_rate', ))
        self.drop_out_initiated = True

    def _create_drop_out(self, shape):
        if self.drop_out_keep_rate and self.drop_out_keep_rate != 1 and self.drop_out_keep_rate > 0:
            if not self.drop_out:
                self.drop_out = DropOut(
                    noise_shape=shape, rate=self.drop_out_keep_rate)
                self.drop_out.set_partition(self.partitions)
                if self.do_backward_select_strategy:
                    self.drop_out.do_backward_select_strategy()

            self.drop_out.generate_mask()

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

    """
    Plaintext forward/backward, these interfaces are for testing
    """

    def plaintext_forward(self, guest_input, epoch=0, batch=0, train=True):

        if self.model is None:
            self.model = recover_sequential_from_dict(self.layer_config)[0]

            if self.float64:
                self.model.type(torch.float64)

        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(
                params=self.model.parameters(), lr=self.learning_rate)

        if train:
            self.model.train()
        else:
            self.model.eval()

        with torch.no_grad():
            guest_tensor = torch.from_numpy(guest_input)
            host_inputs = self.get_forward_from_host(
                epoch, batch, train, idx=-1)
            host_tensors = [torch.from_numpy(arr) for arr in host_inputs]
            interactive_out = self.model(guest_tensor, host_tensors)

        self.guest_tensor = guest_tensor
        self.host_tensors = host_tensors

        return interactive_out.cpu().detach().numpy()

    def plaintext_backward(self, output_gradient, epoch, batch):

        # compute input gradient
        self.guest_tensor: torch.Tensor = self.guest_tensor.requires_grad_(True)
        for tensor in self.host_tensors:
            tensor.requires_grad_(True)
        out = self.model(self.guest_tensor, self.host_tensors)
        loss = backward_loss(out, torch.from_numpy(output_gradient))
        backward_list = [self.guest_tensor]
        backward_list.extend(self.host_tensors)
        ret_grad = autograd.grad(loss, backward_list)

        # update model
        self.guest_tensor: torch.Tensor = self.guest_tensor.requires_grad_(False)
        for tensor in self.host_tensors:
            tensor.requires_grad_(False)
        self.optimizer.zero_grad()
        out = self.model(self.guest_tensor, self.host_tensors)
        loss = backward_loss(out, torch.from_numpy(output_gradient))
        loss.backward()
        self.optimizer.step()

        self.guest_tensor, self.host_tensors = None, None

        for idx, host_grad in enumerate(ret_grad[1:]):
            self.send_host_backward_to_host(host_grad, epoch, batch, idx=idx)

        return ret_grad[0]

    """
    Activation forward & backward
    """

    def activation_forward(self, dense_out, with_grad=True):
        if with_grad:
            if (self.dense_output_data_require_grad is not None) or (
                    self.activation_out_require_grad is not None):
                raise ValueError(
                    'torch forward error, related required grad tensors are not freed')
            self.dense_output_data_require_grad = dense_out.requires_grad_(
                True)
            activation_out_ = self.model.activation(
                self.dense_output_data_require_grad)
            self.activation_out_require_grad = activation_out_
        else:
            with torch.no_grad():
                activation_out_ = self.model.activation(dense_out)

        return activation_out_.cpu().detach().numpy()

    def activation_backward(self, output_gradients):

        if self.activation_out_require_grad is None and self.dense_output_data_require_grad is None:
            raise ValueError('related grad is None, cannot compute backward')
        loss = backward_loss(
            self.activation_out_require_grad,
            torch.Tensor(output_gradients))
        activation_backward_grad = torch.autograd.grad(
            loss, self.dense_output_data_require_grad)
        self.activation_out_require_grad = None
        self.dense_output_data_require_grad = None

        return activation_backward_grad[0].cpu().detach().numpy()

    """
    Forward & Backward
    """

    def print_log(self, descr, epoch, batch, train):
        if train:
            LOGGER.info("{} epoch {} batch {}"
                        "".format(descr, epoch, batch))
        else:
            LOGGER.info("predicting, {} pred iteration {} batch {}"
                        "".format(descr, epoch, batch))

    def forward_interactive(
            self,
            encrypted_host_input,
            epoch,
            batch,
            train=True):

        self.print_log(
            'get encrypted dense output of host model of',
            epoch,
            batch,
            train)
        mask_table_list = []
        guest_nosies = []

        host_idx = 0
        for model, host_bottom_input in zip(
                self.host_model_list, encrypted_host_input):

            encrypted_fw = model(host_bottom_input, self.fixed_point_encoder)
            mask_table = None
            if train:
                self._create_drop_out(encrypted_fw.shape)
                if self.drop_out:
                    mask_table = self.drop_out.generate_mask_table(
                        encrypted_fw.shape)
                if mask_table:
                    encrypted_fw = encrypted_fw.select_columns(mask_table)
                    mask_table_list.append(mask_table)

            guest_forward_noise = self.rng_generator.fast_generate_random_number(
                encrypted_fw.shape, encrypted_fw.partitions, keep_table=mask_table)
            if self.fixed_point_encoder:
                encrypted_fw += guest_forward_noise.encode(
                    self.fixed_point_encoder)
            else:
                encrypted_fw += guest_forward_noise

            guest_nosies.append(guest_forward_noise)

            self.send_guest_encrypted_forward_output_with_noise_to_host(
                encrypted_fw.get_obj(), epoch, batch, idx=host_idx)
            if mask_table:
                self.send_interactive_layer_drop_out_table(
                    mask_table, epoch, batch, idx=host_idx)

            host_idx += 1

        # get list from hosts
        decrypted_dense_outputs = self.get_guest_decrypted_forward_from_host(
            epoch, batch, idx=-1)
        merge_output = None
        for idx, (outputs, noise) in enumerate(
                zip(decrypted_dense_outputs, guest_nosies)):
            out = PaillierTensor(outputs) - noise
            if len(mask_table_list) != 0:
                out = PaillierTensor(
                    out.get_obj().join(
                        mask_table_list[idx],
                        self.expand_columns))
            if merge_output is None:
                merge_output = out
            else:
                merge_output = merge_output + out

        return merge_output

    def forward(self, x, epoch: int, batch: int, train: bool = True, **kwargs):

        self.print_log(
            'interactive layer running forward propagation',
            epoch,
            batch,
            train)
        if self.plaintext:
            return self.plaintext_forward(x, epoch, batch, train)

        if self.model is None:
            self.model = recover_sequential_from_dict(self.layer_config)[0]
            LOGGER.debug('interactive model is {}'.format(self.model))
            # for multi host cases
            LOGGER.debug(
                'host num is {}, len host model {}'.format(
                    self.host_num, len(
                        self.model.host_model)))
            assert self.host_num == len(self.model.host_model), 'host number is {}, but host linear layer number is {},' \
                                                                'please check your interactive configuration, make sure' \
                                                                ' that host layer number equals to host number' \
                .format(self.host_num, len(self.model.host_model))

            if self.float64:
                self.model.type(torch.float64)

        if train and not self.drop_out_initiated:
            self.init_drop_out_parameter()

        host_inputs = self.get_forward_from_host(epoch, batch, train, idx=-1)

        host_bottom_inputs_tensor = []
        host_input_shapes = []
        for i in host_inputs:
            pt = PaillierTensor(i)
            host_bottom_inputs_tensor.append(pt)
            host_input_shapes.append(pt.shape[1])

        self.model.lazy_to_linear(x.shape[1], host_dims=host_input_shapes)
        self.host_input_shapes = host_input_shapes

        if self.guest_model is None:
            LOGGER.info("building interactive layers' training model")
            self._build_model()

        if not self.partitions:
            self.partitions = host_bottom_inputs_tensor[0].partitions

        if not self.send_output_unit:
            self.send_output_unit = True
            for idx in range(self.host_num):
                self.send_interactive_layer_output_unit(
                    self.host_model_list[idx].output_shape[0], idx=idx)

        guest_output = self.guest_model(x)
        host_output = self.forward_interactive(
            host_bottom_inputs_tensor, epoch, batch, train)

        if guest_output is not None:
            dense_output_data = host_output + \
                PaillierTensor(guest_output, partitions=self.partitions)
        else:
            dense_output_data = host_output

        self.print_log(
            "start to get interactive layer's activation output of",
            epoch,
            batch,
            train)

        if self.float64:  # result after encrypt calculation is float 64
            dense_out = torch.from_numpy(dense_output_data.numpy())
        else:
            dense_out = torch.Tensor(
                dense_output_data.numpy())  # convert to float32

        if self.do_backward_select_strategy:
            for h in self.host_model_list:
                h.activation_input = dense_out.cpu().detach().numpy()

        # if is not backward strategy, can compute grad directly
        if not train or self.do_backward_select_strategy:
            with_grad = False
        else:
            with_grad = True

        activation_out = self.activation_forward(
            dense_out, with_grad=with_grad)

        if train and self.drop_out:
            return self.drop_out.forward(activation_out)

        return activation_out

    def backward_interactive(
            self,
            host_model,
            activation_gradient,
            epoch,
            batch,
            host_idx):

        LOGGER.info(
            "get encrypted weight gradient of epoch {} batch {}".format(
                epoch, batch))
        encrypted_weight_gradient = host_model.get_weight_gradient(
            activation_gradient, encoder=self.fixed_point_encoder)
        if self.fixed_point_encoder:
            encrypted_weight_gradient = self.fixed_point_encoder.decode(
                encrypted_weight_gradient)
        noise_w = self.rng_generator.generate_random_number(
            encrypted_weight_gradient.shape)
        self.transfer_variable.encrypted_guest_weight_gradient.remote(
            encrypted_weight_gradient +
            noise_w,
            role=consts.HOST,
            idx=host_idx,
            suffix=(
                epoch,
                batch,
            ))
        LOGGER.info(
            "get decrypted weight graident of epoch {} batch {}".format(
                epoch, batch))
        decrypted_weight_gradient = self.transfer_variable.decrypted_guest_weight_gradient.get(
            idx=host_idx, suffix=(epoch, batch,))
        decrypted_weight_gradient -= noise_w
        encrypted_acc_noise = self.get_encrypted_acc_noise_from_host(
            epoch, batch, idx=host_idx)

        return decrypted_weight_gradient, encrypted_acc_noise

    def backward(self, error, epoch: int, batch: int, selective_ids=None):

        if self.plaintext:
            return self.plaintext_backward(error, epoch, batch)

        if selective_ids:

            for host_model in self.host_model_list:
                host_model.select_backward_sample(selective_ids)
            self.guest_model.select_backward_sample(selective_ids)

            if self.drop_out:
                self.drop_out.select_backward_sample(selective_ids)

        if self.do_backward_select_strategy:
            # send to all host
            self.send_backward_select_info(
                selective_ids, len(error), epoch, batch, -1)

        if len(error) > 0:

            LOGGER.debug(
                "interactive layer start backward propagation of epoch {} batch {}".format(
                    epoch, batch))
            if not self.do_backward_select_strategy:
                activation_gradient = self.activation_backward(error)
            else:
                act_input = self.host_model_list[0].get_selective_activation_input(
                )
                _ = self.activation_forward(torch.from_numpy(act_input), True)
                activation_gradient = self.activation_backward(error)

            if self.drop_out:
                activation_gradient = self.drop_out.backward(
                    activation_gradient)
            LOGGER.debug(
                "interactive layer update guest weight of epoch {} batch {}".format(
                    epoch, batch))

            # update guest model
            guest_input_gradient = self.update_guest(activation_gradient)

            LOGGER.debug('update host model weights')
            for idx, host_model in enumerate(self.host_model_list):
                # update host models
                host_weight_gradient, acc_noise = self.backward_interactive(
                    host_model, activation_gradient, epoch, batch, host_idx=idx)
                host_input_gradient = self.update_host(
                    host_model, activation_gradient, host_weight_gradient, acc_noise)
                self.send_host_backward_to_host(
                    host_input_gradient.get_obj(), epoch, batch, idx=idx)

            return guest_input_gradient
        else:
            return []

    """
    Model update
    """

    def update_guest(self, activation_gradient):
        input_gradient = self.guest_model.get_input_gradient(
            activation_gradient)
        weight_gradient = self.guest_model.get_weight_gradient(
            activation_gradient)
        self.guest_model.update_weight(weight_gradient)
        self.guest_model.update_bias(activation_gradient)

        return input_gradient

    def update_host(
            self,
            host_model,
            activation_gradient,
            weight_gradient,
            acc_noise):
        activation_gradient_tensor = PaillierTensor(
            activation_gradient, partitions=self.partitions)
        input_gradient = host_model.get_input_gradient(
            activation_gradient_tensor, acc_noise, encoder=self.fixed_point_encoder)
        host_model.update_weight(weight_gradient)
        host_model.update_bias(activation_gradient)

        return input_gradient

    """
    Communication functions
    """

    def send_interactive_layer_output_unit(self, shape, idx=0):
        self.transfer_variable.interactive_layer_output_unit.remote(
            shape, role=consts.HOST, idx=idx)

    def send_backward_select_info(
            self,
            selective_ids,
            gradient_len,
            epoch,
            batch,
            idx):
        self.transfer_variable.selective_info.remote(
            (selective_ids, gradient_len), role=consts.HOST, idx=idx, suffix=(
                epoch, batch,))

    def send_host_backward_to_host(self, host_error, epoch, batch, idx):
        self.transfer_variable.host_backward.remote(host_error,
                                                    role=consts.HOST,
                                                    idx=idx,
                                                    suffix=(epoch, batch,))

    def get_forward_from_host(self, epoch, batch, train, idx=0):
        return self.transfer_variable.encrypted_host_forward.get(
            idx=idx, suffix=(epoch, batch, train))

    def send_guest_encrypted_forward_output_with_noise_to_host(
            self, encrypted_guest_forward_with_noise, epoch, batch, idx):
        return self.transfer_variable.encrypted_guest_forward.remote(
            encrypted_guest_forward_with_noise,
            role=consts.HOST,
            idx=idx,
            suffix=(
                epoch,
                batch,
            ))

    def send_interactive_layer_drop_out_table(
            self, mask_table, epoch, batch, idx):
        return self.transfer_variable.drop_out_table.remote(
            mask_table, role=consts.HOST, idx=idx, suffix=(epoch, batch,))

    def get_guest_decrypted_forward_from_host(self, epoch, batch, idx=0):
        return self.transfer_variable.decrypted_guest_forward.get(
            idx=idx, suffix=(epoch, batch,))

    def get_encrypted_acc_noise_from_host(self, epoch, batch, idx=0):
        return self.transfer_variable.encrypted_acc_noise.get(
            idx=idx, suffix=(epoch, batch,))

    """
    Model IO
    """

    def transfer_np_model_to_torch_interactive_layer(self):

        self.model = self.model.cpu()

        if self.guest_model is not None:
            guest_weight = self.guest_model.get_weight()
            model: torch.nn.Linear = self.model.guest_model
            model.weight.data.copy_(torch.Tensor(guest_weight))
            if self.guest_model.bias is not None:
                model.bias.data.copy_(torch.Tensor(self.guest_model.bias))

        for host_np_model, torch_model in zip(
                self.host_model_list, self.model.host_model):
            host_weight = host_np_model.get_weight()
            torch_model.weight.data.copy_(torch.Tensor(host_weight))
            if host_np_model.bias is not None:
                torch_model.bias.data.copy_(torch.Tensor(torch_model.bias))

    def export_model(self):

        self.transfer_np_model_to_torch_interactive_layer()
        interactive_layer_param = InteractiveLayerParam()
        interactive_layer_param.interactive_guest_saved_model_bytes = TorchNNModel.get_model_bytes(
            self.model)
        interactive_layer_param.host_input_shape.extend(self.host_input_shapes)

        return interactive_layer_param

    def restore_model(self, interactive_layer_param):

        self.host_input_shapes = list(interactive_layer_param.host_input_shape)
        self.model = TorchNNModel.recover_model_bytes(
            interactive_layer_param.interactive_guest_saved_model_bytes)
        self._build_model()


class HEInteractiveLayerHost(InteractiveLayerHost):

    def __init__(self, params):
        super(HEInteractiveLayerHost, self).__init__(params)

        self.plaintext = PLAINTEXT
        self.acc_noise = None
        self.learning_rate = params.interactive_layer_lr
        self.encrypter = self.generate_encrypter(params)
        self.transfer_variable = HEInteractiveTransferVariable()
        self.partitions = 1
        self.input_shape = None
        self.output_unit = None
        self.rng_generator = RandomNumberGenerator()
        self.do_backward_select_strategy = False
        self.drop_out_init = False
        self.drop_out_keep_rate = None
        self.fixed_point_encoder = None if params.floating_point_precision is None else FixedPointEncoder(
            2 ** params.floating_point_precision)
        self.mask_table = None

    """
    Init
    """

    def set_transfer_variable(self, transfer_variable):
        self.transfer_variable = transfer_variable

    def set_partition(self, partition):
        self.partitions = partition

    def set_backward_select_strategy(self):
        self.do_backward_select_strategy = True

    """
    Forward & Backward
    """

    def plaintext_forward(self, host_input, epoch, batch, train):
        self.send_forward_to_guest(host_input, epoch, batch, train)

    def plaintext_backward(self, epoch, batch):
        return self.get_host_backward_from_guest(epoch, batch)

    def forward(self, host_input, epoch=0, batch=0, train=True, **kwargs):

        if self.plaintext:
            self.plaintext_forward(host_input, epoch, batch, train)
            return

        if train and not self.drop_out_init:
            self.drop_out_init = True
            self.drop_out_keep_rate = self.transfer_variable.drop_out_info.get(
                0, role=consts.GUEST, suffix=('dropout_rate', ))
            if self.drop_out_keep_rate == -1:
                self.drop_out_keep_rate = None

        LOGGER.info(
            "forward propagation: encrypt host_bottom_output of epoch {} batch {}".format(
                epoch, batch))
        host_input = PaillierTensor(host_input, partitions=self.partitions)

        encrypted_host_input = host_input.encrypt(self.encrypter)
        self.send_forward_to_guest(
            encrypted_host_input.get_obj(), epoch, batch, train)

        encrypted_guest_forward = PaillierTensor(
            self.get_guest_encrypted_forward_from_guest(epoch, batch))
        decrypted_guest_forward = encrypted_guest_forward.decrypt(
            self.encrypter)

        if self.fixed_point_encoder:
            decrypted_guest_forward = decrypted_guest_forward.decode(
                self.fixed_point_encoder)

        if self.input_shape is None:
            self.input_shape = host_input.shape[1]
            self.output_unit = self.get_interactive_layer_output_unit()

        if self.acc_noise is None:
            self.acc_noise = np.zeros((self.input_shape, self.output_unit))

        mask_table = None
        if train and self.drop_out_keep_rate and self.drop_out_keep_rate < 1:
            mask_table = self.get_interactive_layer_drop_out_table(
                epoch, batch)

        if mask_table:
            decrypted_guest_forward_with_noise = decrypted_guest_forward + \
                (host_input * self.acc_noise).select_columns(mask_table)
            self.mask_table = mask_table
        else:
            noise_part = (host_input * self.acc_noise)
            decrypted_guest_forward_with_noise = decrypted_guest_forward + noise_part

        self.send_decrypted_guest_forward_with_noise_to_guest(
            decrypted_guest_forward_with_noise.get_obj(), epoch, batch)

    def backward(self, epoch, batch):

        if self.plaintext:
            return self.plaintext_backward(epoch, batch), []

        do_backward = True
        selective_ids = []
        if self.do_backward_select_strategy:
            selective_ids, do_backward = self.send_backward_select_info(
                epoch, batch)

        if not do_backward:
            return [], selective_ids

        encrypted_guest_weight_gradient = self.get_guest_encrypted_weight_gradient_from_guest(
            epoch, batch)

        LOGGER.info(
            "decrypt weight gradient of epoch {} batch {}".format(
                epoch, batch))
        decrypted_guest_weight_gradient = self.encrypter.recursive_decrypt(
            encrypted_guest_weight_gradient)
        noise_weight_gradient = self.rng_generator.generate_random_number(
            (self.input_shape, self.output_unit))
        decrypted_guest_weight_gradient += noise_weight_gradient / self.learning_rate
        self.send_guest_decrypted_weight_gradient_to_guest(
            decrypted_guest_weight_gradient, epoch, batch)
        LOGGER.info(
            "encrypt acc_noise of epoch {} batch {}".format(
                epoch, batch))
        encrypted_acc_noise = self.encrypter.recursive_encrypt(self.acc_noise)
        self.send_encrypted_acc_noise_to_guest(
            encrypted_acc_noise, epoch, batch)
        self.acc_noise += noise_weight_gradient
        host_input_gradient = PaillierTensor(
            self.get_host_backward_from_guest(epoch, batch))
        host_input_gradient = host_input_gradient.decrypt(self.encrypter)

        if self.fixed_point_encoder:
            host_input_gradient = host_input_gradient.decode(
                self.fixed_point_encoder).numpy()
        else:
            host_input_gradient = host_input_gradient.numpy()

        return host_input_gradient, selective_ids

    """
    Communication Function
    """

    def send_backward_select_info(self, epoch, batch):
        selective_ids, do_backward = self.transfer_variable.selective_info.get(
            idx=0, suffix=(epoch, batch,))

        return selective_ids, do_backward

    def send_encrypted_acc_noise_to_guest(
            self, encrypted_acc_noise, epoch, batch):
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
        return self.transfer_variable.drop_out_table.get(
            idx=0, suffix=(epoch, batch,))

    def send_forward_to_guest(self, encrypted_host_input, epoch, batch, train):
        self.transfer_variable.encrypted_host_forward.remote(
            encrypted_host_input, idx=0, role=consts.GUEST, suffix=(epoch, batch, train))

    def send_guest_decrypted_weight_gradient_to_guest(
            self, decrypted_guest_weight_gradient, epoch, batch):
        self.transfer_variable.decrypted_guest_weight_gradient.remote(
            decrypted_guest_weight_gradient, idx=0, role=consts.GUEST, suffix=(epoch, batch,))

    def get_host_backward_from_guest(self, epoch, batch):
        host_backward = self.transfer_variable.host_backward.get(
            idx=0, suffix=(epoch, batch,))

        return host_backward

    def get_guest_encrypted_forward_from_guest(self, epoch, batch):
        encrypted_guest_forward = self.transfer_variable.encrypted_guest_forward.get(
            idx=0, suffix=(epoch, batch,))

        return encrypted_guest_forward

    def send_decrypted_guest_forward_with_noise_to_guest(
            self, decrypted_guest_forward_with_noise, epoch, batch):
        self.transfer_variable.decrypted_guest_forward.remote(
            decrypted_guest_forward_with_noise,
            idx=0,
            role=consts.GUEST,
            suffix=(
                epoch,
                batch,
            ))

    """
    Encrypter
    """

    def generate_encrypter(self, param):
        LOGGER.info("generate encrypter")
        if param.encrypt_param.method.lower() == consts.PAILLIER.lower():
            encrypter = PaillierEncrypt()
            encrypter.generate_key(param.encrypt_param.key_length)
        else:
            raise NotImplementedError("encrypt method not supported yet!!!")

        return encrypter

    """
    Model IO
    """

    def export_model(self):
        interactive_layer_param = InteractiveLayerParam()
        interactive_layer_param.acc_noise = pickle.dumps(self.acc_noise)

        return interactive_layer_param

    def restore_model(self, interactive_layer_param):
        self.acc_noise = pickle.loads(interactive_layer_param.acc_noise)
