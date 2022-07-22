import numpy as np
import tempfile
from federatedml.util import LOGGER

try:  # for the situation that torch is not installed, but other modules still can be used
    import torch
    import torch as t
    from torch import autograd
    from federatedml.nn.backend.fate_torch import optim, serialization as s
    from federatedml.nn.backend.fate_torch.base import FateTorchOptimizer
    from federatedml.nn.backend.fate_torch.nn import CrossEntropyLoss
except ImportError:
    pass


def backward_loss(z, backward_error):
    return t.sum(z * backward_error)


class PytorchDataConvertor(object):

    def __init__(self,):
        pass

    def convert_data(self, x: np.ndarray, y: np.ndarray = None):

        tensorx = t.Tensor(x)
        if y is not None:
            tensory = t.Tensor(y)
            return tensorx, tensory
        else:
            return tensorx


def label_convert(y, loss_fn):

    # pytorch CE loss require 1D-int64-tensor
    if isinstance(loss_fn, CrossEntropyLoss):
        return t.Tensor(y).flatten().type(t.int64).flatten()  # accept 1-D array
    else:
        return t.Tensor(y).type(t.float)


class PytorchNNModel(object):

    def __init__(self, nn_define: dict, optimizer_define: dict = None, loss_fn_define: dict = None):

        self.model = s.recover_sequential_from_dict(nn_define)
        if optimizer_define is None:  # default optimizer
            self.optimizer = optim.SGD(lr=0.01)
        else:
            self.optimizer: FateTorchOptimizer = s.recover_optimizer_from_dict(optimizer_define)
        self.opt_inst = self.optimizer.to_torch_instance(self.model.parameters())

        if loss_fn_define is None:
            self.loss_fn = backward_loss
        else:
            self.loss_fn = s.recover_loss_fn_from_dict(loss_fn_define)
            self.loss_define = loss_fn_define

        self.forward_cache: t.Tensor = None
        self.train_mode: bool = True
        self.loss_history = []

        self.x_dtype = None
        self.y_dtype = None

    def print_parameters(self):
        LOGGER.debug('model parameter is {}'.format(list(self.model.parameters())))

    def __repr__(self):
        return self.model.__repr__() + '\n' + self.optimizer.__repr__() + '\n' + str(self.loss_fn)

    def input_tensor_convert(self, data):
        if self.x_dtype is not None:
            return torch.tensor(data, dtype=self.x_dtype)
        else:
            pass

    def label_tensor_convert(self, data):
        if self.y_dtype is not None:
            return torch.tensor(data, dtype=self.y_dtype)
        else:
            pass

    def train(self, data_x_and_y, ret_input_gradient=False):

        x, y = data_x_and_y  # this is a tuple
        input_grad = None
        self.opt_inst.zero_grad()
        yt = label_convert(y, self.loss_fn)
        if ret_input_gradient:
            xt = t.Tensor(x).requires_grad_(True)
        else:
            xt = t.Tensor(x)

        loss = self.loss_fn(self.model(xt), yt)

        if ret_input_gradient:
            input_grad = autograd.grad(loss, xt, retain_graph=True)

        loss.backward()
        self.loss_history.append(loss.detach().numpy())
        self.opt_inst.step()

        if ret_input_gradient:
            return loss.detach().numpy(), input_grad
        else:
            return loss.detach().numpy()

    def forward(self, x):

        x = t.Tensor(x)
        forward_rs = self.model(x)
        self.forward_cache = forward_rs

        return forward_rs.detach().numpy()

    def backward(self, backward_gradients):

        if self.forward_cache is None:
            raise ValueError('no forward cache, unable to do backward propagation')
        self.forward_cache.backward(t.Tensor(backward_gradients))

    def backward_and_update(self, backward_gradients):

        self.opt_inst.zero_grad()
        if self.forward_cache is None:
            raise ValueError('no forward cache, unable to do backward propagation')
        self.forward_cache.backward(t.Tensor(backward_gradients))
        self.opt_inst.step()

    def predict(self, x):
        return self.model(t.Tensor(x)).detach().numpy()

    def get_forward_loss_from_input(self, x, y, reduction='none'):

        with torch.no_grad():
            default_reduction = self.loss_fn.reduction
            self.loss_fn.reduction = reduction
            yt = label_convert(y, self.loss_fn)
            xt = t.Tensor(x)
            loss = self.loss_fn(self.model(xt), yt)
            self.loss_fn.reduction = default_reduction

        return list(map(float, loss.detach().numpy()))

    def get_input_gradients(self, x, y):

        yt = label_convert(y, self.loss_fn)
        xt = t.Tensor(x).requires_grad_(True)
        fw = self.model(xt)
        loss = self.loss_fn(fw, yt)
        grad = autograd.grad(loss, xt)

        return (grad[0].detach().numpy(), )

    def get_loss(self):
        return [self.loss_history[-1]]

    def get_trainable_gradients(self, x, y):
        pass

    def evaluate(self, data):
        pass

    @staticmethod
    def get_model_bytes(model):
        with tempfile.TemporaryFile() as f:
            torch.save(model, f)
            f.seek(0)
            return f.read()

    def export_model(self):
        return self.get_model_bytes(self.model)

    @staticmethod
    def recover_model_bytes(model_bytes):
        with tempfile.TemporaryFile() as f:
            f.write(model_bytes)
            f.seek(0)
            model = torch.load(f)
        return model

    def restore_model(self, model_bytes):
        self.model = self.recover_model_bytes(model_bytes)
        LOGGER.debug('loaded model is {}'.format(self.model))
        return self

    def compile(self, *args, **kwargs):
        pass
