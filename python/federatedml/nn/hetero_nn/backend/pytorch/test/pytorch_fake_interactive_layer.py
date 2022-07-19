from federatedml.nn.backend.fate_torch import recover_sequential_from_dict
from federatedml.nn.hetero_nn.backend.pytorch.pytorch_nn_model import PytorchNNModel
from federatedml.nn.backend.fate_torch import Sequential
from federatedml.nn.backend.fate_torch import SGD
from federatedml.nn.backend.fate_torch import Linear
from torch import optim
from federatedml.util import LOGGER, consts
import torch as t


class FakeInteractiveLayerGuest(object):

    def __init__(self, params=None, layer_config=None):

        self.nn_define = layer_config
        self.layer_config = layer_config
        self.model = None
        self.transfer_variable = None
        self.learning_rate = params.interactive_layer_lr

        self.guest_input: t.Tensor = None
        self.host_input: t.Tensor = None
        self.guest_output = None
        self.host_output: t.Tensor = None

        self.dense_output_data: t.Tensor = None

        self.guest_model = None
        self.host_model = None

        self.partitions = 0
        self.do_backward_select_strategy = False
        self.encrypted_host_input_cached = None
        self.drop_out_keep_rate = params.drop_out_keep_rate
        self.drop_out = None
        self.sync_output_unit = False

        self.guest_opt, self.host_opt = None, None

        tmp_model = recover_sequential_from_dict(self.layer_config)
        self.guest_model = Sequential(Linear(in_features=tmp_model[0].in_features,
                                             out_features=tmp_model[0].out_features,
                                             bias=False))
        self.guest_model[0].weight = t.nn.Parameter(tmp_model[0].weight.detach())
        self.guest_opt = optim.SGD(self.guest_model.parameters(), lr=self.learning_rate)

        self.host_model = recover_sequential_from_dict(self.layer_config)
        self.host_opt = optim.SGD(self.host_model.parameters(), lr=self.learning_rate)

        self.local_mode = False

        self.print_parameter()

        LOGGER.debug('running faking')

    def zero_grad(self):
        self.guest_opt.zero_grad()
        self.host_opt.zero_grad()

    def local_fw(self, guest_input, host_input):
        return self.guest_model(guest_input) + self.host_model(host_input)

    def update(self):
        self.host_opt.step()
        self.guest_opt.step()

    def print_parameter(self):
        LOGGER.debug('guest model {}'.format(list(self.guest_model.parameters())))
        LOGGER.debug('host model {}'.format(list(self.host_model.parameters())))

    def forward(self, guest_input, epoch=0, batch=0, train=True, host_input=None):

        xt = t.Tensor(guest_input).requires_grad_(True)
        guest_forward = self.guest_model.forward(xt)

        if not self.local_mode:
            host_input = self.transfer_variable.encrypted_host_forward.get(idx=0, suffix=(epoch, batch,))

        xt2 = t.Tensor(host_input).requires_grad_(True)
        host_forward = self.host_model.forward(xt2)
        rs = guest_forward + host_forward

        if train:
            if self.guest_input is None and self.host_input is None:
                self.guest_input = xt
                self.host_input = xt2
            if self.dense_output_data is None:
                self.dense_output_data = rs

        return rs.detach().numpy()

    def backward(self, output_gradient, selective_ids=None, epoch=0, batch=0):

        # backwards
        self.guest_opt.zero_grad()
        self.host_opt.zero_grad()
        LOGGER.debug('output gradient is {}'.format(output_gradient))

        self.dense_output_data.backward(t.Tensor(output_gradient))
        self.guest_opt.step()
        self.host_opt.step()

        guest_input_grad = self.guest_input.grad.detach().numpy()
        host_input_grad = self.host_input.grad.detach().numpy()
        self.dense_output_data = None

        self.guest_input = None
        self.host_input = None

        if not self.local_mode:
            self.transfer_variable.host_backward.remote(host_input_grad,
                                                        role=consts.HOST,
                                                        idx=0,
                                                        suffix=(epoch, batch,))

            return guest_input_grad
        else:
            return guest_input_grad, host_input_grad

    def set_transfer_variable(self, trans_var):
        self.transfer_variable = trans_var

    def set_backward_select_strategy(self):
        pass

    def set_partition(self, partition):
        pass


class FakeInteractiveLayerHost(object):

    def __init__(self, params=None, layer_config=None):

        self.nn_define = layer_config
        self.layer_config = layer_config
        self.model = None
        self.transfer_variable = None
        self.learning_rate = params.interactive_layer_lr

        self.guest_input = None
        self.guest_output = None
        self.host_output = None

        self.dense_output_data = None

        self.guest_model = None
        self.host_model = None

        self.partitions = 0
        self.do_backward_select_strategy = False
        self.encrypted_host_input_cached = None
        self.drop_out_keep_rate = params.drop_out_keep_rate
        self.drop_out = None
        self.sync_output_unit = False

        self.optimizer = SGD(lr=self.learning_rate)

    def forward(self, host_input, epoch=0, batch=0, train=True):

        self.transfer_variable.encrypted_host_forward.remote(host_input, suffix=(epoch, batch,))

    def backward(self, epoch, batch,):

        return self.transfer_variable.host_backward.get(suffix=(epoch, batch,)), None

    def set_partition(self, partition):
        pass

    def set_transfer_variable(self, trans_var):
        self.transfer_variable = trans_var
