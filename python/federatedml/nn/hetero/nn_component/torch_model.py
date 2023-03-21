import numpy as np
import tempfile
from federatedml.util import LOGGER

try:  # for the situation that torch is not installed, but other modules still can be used
    import torch
    import torch as t
    import copy
    from types import SimpleNamespace
    from torch import autograd
    from federatedml.nn.backend.torch import serialization as s
    from federatedml.nn.backend.torch.base import FateTorchOptimizer
    from federatedml.nn.backend.torch.nn import CrossEntropyLoss
    from federatedml.nn.backend.torch import optim
except ImportError:
    pass


def backward_loss(z, backward_error):
    return t.sum(z * backward_error)


class TorchNNModel(object):

    def __init__(self, nn_define: dict, optimizer_define: dict = None, loss_fn_define: dict = None, cuda=False):

        self.cuda = cuda
        self.double_model = False
        if self.cuda and not t.cuda.is_available():
            raise ValueError(
                'this machine dose not support cuda, cuda.is_available() is False')
        self.optimizer_define = optimizer_define
        self.nn_define = nn_define
        self.loss_fn_define = loss_fn_define
        self.loss_history = []
        self.model, self.opt_inst, self.loss_fn = self.init(
            self.nn_define, self.optimizer_define, self.loss_fn_define)
        self.fw_cached = None

    def to_tensor(self, x: np.ndarray):

        if isinstance(x, np.ndarray):
            x = t.from_numpy(x)

        if self.cuda:
            return x.cuda()
        else:
            return x

    def label_convert(self, y, loss_fn):
        # pytorch CE loss require 1D-int64-tensor
        if isinstance(loss_fn, CrossEntropyLoss):
            return t.Tensor(y).flatten().type(
                t.int64).flatten()  # accept 1-D array
        else:
            return t.Tensor(y).type(t.float)

    def init(self, nn_define: dict, optimizer_define: dict = None, loss_fn_define: dict = None):

        model = s.recover_sequential_from_dict(nn_define)
        if self.cuda:
            model = model.cuda()

        if optimizer_define is None:  # default optimizer
            optimizer = optim.SGD(lr=0.01)
        else:
            optimizer: FateTorchOptimizer = s.recover_optimizer_from_dict(optimizer_define)
        opt_inst = optimizer.to_torch_instance(model.parameters())

        if loss_fn_define is None:
            loss_fn = backward_loss
        else:
            loss_fn = s.recover_loss_fn_from_dict(loss_fn_define)

        if self.double_model:
            self.model.type(t.float64)

        return model, opt_inst, loss_fn

    def print_parameters(self):
        LOGGER.debug(
            'model parameter is {}'.format(
                list(
                    self.model.parameters())))

    def __repr__(self):
        return self.model.__repr__() + '\n' + self.opt_inst.__repr__() + \
            '\n' + str(self.loss_fn)

    def train_mode(self, mode):
        self.model.train(mode)

    def train(self, data_x_and_y):

        x, y = data_x_and_y  # this is a tuple
        self.opt_inst.zero_grad()
        yt = self.to_tensor(y)
        xt = self.to_tensor(x)
        out = self.model(xt)
        loss = self.loss_fn(out, yt)
        loss.backward()
        loss_val = loss.cpu().detach().numpy()
        self.loss_history.append(loss_val)
        self.opt_inst.step()

        return loss_val

    def forward(self, x):
        # will cache tensor with grad, this function is especially for bottom
        # model
        x = self.to_tensor(x)
        out = self.model(x)
        if self.fw_cached is not None:
            raise ValueError('fed cached should be None when forward')
        self.fw_cached = out

        return out.cpu().detach().numpy()

    def backward(self, error):
        # backward ,this function is especially for bottom model
        self.opt_inst.zero_grad()
        error = self.to_tensor(error)
        loss = self.loss_fn(self.fw_cached, error)
        loss.backward()
        self.fw_cached = None
        self.opt_inst.step()

    def predict(self, x):

        with torch.no_grad():
            return self.model(self.to_tensor(x)).cpu().detach().numpy()

    def get_forward_loss_from_input(self, x, y, reduction='none'):

        with torch.no_grad():
            default_reduction = self.loss_fn.reduction
            self.loss_fn.reduction = reduction
            yt = self.to_tensor(y)
            xt = self.to_tensor(x)
            loss = self.loss_fn(self.model(xt), yt)
            self.loss_fn.reduction = default_reduction

        return list(map(float, loss.detach().numpy()))

    def get_input_gradients(self, x, y):

        yt = self.to_tensor(y)
        xt = self.to_tensor(x).requires_grad_(True)
        fw = self.model(xt)
        loss = self.loss_fn(fw, yt)
        grad = autograd.grad(loss, xt)

        return [grad[0].detach().numpy()]

    def get_loss(self):
        return [self.loss_history[-1]]

    @staticmethod
    def get_model_bytes(model):
        with tempfile.TemporaryFile() as f:
            torch.save(model, f)
            f.seek(0)
            return f.read()

    @staticmethod
    def recover_model_bytes(model_bytes):
        with tempfile.TemporaryFile() as f:
            f.write(model_bytes)
            f.seek(0)
            model = torch.load(f)
        return model

    @staticmethod
    def get_model_save_dict(model: t.nn.Module, model_define, optimizer: t.optim.Optimizer, optimizer_define,
                            loss_define):
        with tempfile.TemporaryFile() as f:
            save_dict = {
                'nn_define': model_define,
                'model': model.state_dict(),
                'optimizer_define': optimizer_define,
                'optimizer': optimizer.state_dict(),
                'loss_define': loss_define
            }
            torch.save(save_dict, f)
            f.seek(0)
            return f.read()

    @staticmethod
    def recover_model_save_dict(model_bytes):
        with tempfile.TemporaryFile() as f:
            f.write(model_bytes)
            f.seek(0)
            save_dict = torch.load(f)

        return save_dict

    def restore_model(self, model_bytes):
        save_dict = self.recover_model_save_dict(model_bytes)
        self.nn_define = save_dict['nn_define']
        opt_define = save_dict['optimizer_define']
        # optimizer can be updated
        # old define == new define, load state dict
        if opt_define == self.optimizer_define:
            opt_inst: t.optim.Optimizer = self.opt_inst
            opt_inst.load_state_dict(save_dict['optimizer'])
        # load state dict
        self.model.load_state_dict(save_dict['model'])

        return self

    def export_model(self):
        return self.get_model_save_dict(
            self.model,
            self.nn_define,
            self.opt_inst,
            self.optimizer_define,
            self.loss_fn_define)
