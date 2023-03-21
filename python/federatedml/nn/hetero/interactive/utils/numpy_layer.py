import torch
import numpy as np
from federatedml.util import consts
from federatedml.secureprotol.paillier_tensor import PaillierTensor


class NumpyDenseLayer(object):

    """
    NumpyDenseLayer is designed for Pailler Tensor compute
    """

    def __init__(self):

        self.input = None
        self.model_weight = None
        self.model_shape = None
        self.bias = None
        self.lr = 1.0
        self.role = None
        self.is_empty_model = False
        self.activation_input = None
        self.input_cached = np.array([])
        self.activation_cached = np.array([])
        self.do_backward_selective_strategy = False
        self.batch_size = None

    def set_backward_selective_strategy(self):
        self.do_backward_selective_strategy = True

    def set_batch(self, batch_size):
        self.batch_size = batch_size

    def build(self, torch_linear: torch.nn.Linear):

        if torch_linear is None:
            if self.role == "host":
                raise ValueError("host input is empty!")
            self.is_empty_model = True
            return

        assert isinstance(
            torch_linear, torch.nn.Linear), 'must use a torch Linear to build this class, but got {}' .format(torch_linear)

        self.model_weight = torch_linear.weight.cpu().detach().numpy().transpose()
        if torch_linear.bias is not None:
            self.bias = torch_linear.bias.cpu().detach().numpy()

    def export_model(self):
        if self.is_empty_model:
            return "".encode()
        layer_weights = [self.model_weight]
        return layer_weights

    def get_selective_activation_input(self):
        self.activation_input = self.activation_cached[: self.batch_size]
        self.activation_cached = self.activation_cached[self.batch_size:]
        return self.activation_input

    def get_weight(self):
        return self.model_weight.transpose()

    def get_bias(self):
        return self.bias

    def set_learning_rate(self, lr):
        self.lr = lr

    def forward(self, x, **kwargs):
        pass

    def get_weight_gradient(self, delta):
        pass

    def restore_model(self, model_bytes):
        pass

    def update_weight(self, delta):
        pass

    def update_bias(self, delta):
        pass

    @property
    def empty(self):
        return self.is_empty_model

    @property
    def output_shape(self):
        return self.model_weight.shape[1:]

    def __repr__(self):
        return 'model weights: {}, model bias {}'.format(
            self.model_weight, self.bias)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class NumpyDenseLayerGuest(NumpyDenseLayer):

    def __init__(self):
        super(NumpyDenseLayerGuest, self).__init__()
        self.role = consts.GUEST

    def forward(self, x):

        if self.empty:
            return None

        self.input = x
        output = np.matmul(x, self.model_weight)

        if self.bias is not None:
            output += self.bias

        return output

    def select_backward_sample(self, selective_ids):
        if self.input_cached.shape[0] == 0:
            self.input_cached = self.input[selective_ids]
        else:
            self.input_cached = np.vstack(
                (self.input_cached, self.input[selective_ids])
            )

    def get_input_gradient(self, delta):

        if self.empty:
            return None
        error = np.matmul(delta, self.model_weight.T)

        return error

    def get_weight_gradient(self, delta):

        if self.empty:
            return None
        if self.do_backward_selective_strategy:
            self.input = self.input_cached[: self.batch_size]
            self.input_cached = self.input_cached[self.batch_size:]
        delta_w = np.matmul(delta.T, self.input)

        return delta_w

    def update_weight(self, delta):

        if self.empty:
            return None
        self.model_weight -= self.lr * delta.T

    def update_bias(self, delta):
        if self.bias is not None:
            self.bias -= np.sum(delta, axis=0) * self.lr


class NumpyDenseLayerHost(NumpyDenseLayer):

    """
    This dense layer can directly compute pallier-tensor forward
    """

    def __init__(self):
        super(NumpyDenseLayerHost, self).__init__()
        self.role = consts.HOST

    def select_backward_sample(self, selective_ids):

        cached_shape = self.input_cached.shape[0]
        offsets = [i + cached_shape for i in range(len(selective_ids))]
        id_map = dict(zip(selective_ids, offsets))
        if cached_shape == 0:
            self.input_cached = (
                self.input.get_obj()
                .filter(lambda k, v: k in id_map)
                .map(lambda k, v: (id_map[k], v))
            )
            self.input_cached = PaillierTensor(self.input_cached)
            self.activation_cached = self.activation_input[selective_ids]
        else:

            selective_input = (
                self.input.get_obj()
                .filter(lambda k, v: k in id_map)
                .map(lambda k, v: (id_map[k], v))
            )
            self.input_cached = PaillierTensor(
                self.input_cached.get_obj().union(selective_input)
            )
            self.activation_cached = np.vstack(
                (self.activation_cached, self.activation_input[selective_ids])
            )

    def forward(self, x, encoder=None):

        self.input = x
        if encoder is not None:
            output = x * encoder.encode(self.model_weight)
        else:
            output = x * self.model_weight
        if self.bias is not None:
            if encoder is not None:
                output += encoder.encode(self.bias)
            else:
                output += self.bias

        return output

    def get_input_gradient(self, delta, acc_noise, encoder=None):
        if not encoder:
            error = delta * self.model_weight.T + delta * acc_noise.T
        else:
            error = delta.encode(encoder) * (self.model_weight + acc_noise).T

        return error

    def get_weight_gradient(self, delta, encoder=None):

        if self.do_backward_selective_strategy:
            batch_size = self.batch_size
            self.input = PaillierTensor(
                self.input_cached.get_obj().filter(lambda k, v: k < batch_size)
            )
            self.input_cached = PaillierTensor(
                self.input_cached.get_obj()
                .filter(lambda k, v: k >= batch_size)
                .map(lambda k, v: (k - batch_size, v))
            )

        if encoder:
            delta_w = self.input.fast_matmul_2d(encoder.encode(delta))
        else:
            delta_w = self.input.fast_matmul_2d(delta)

        return delta_w

    def update_weight(self, delta):
        self.model_weight -= delta * self.lr

    def update_bias(self, delta):
        if self.bias is not None:
            self.bias -= np.sum(delta, axis=0) * self.lr
