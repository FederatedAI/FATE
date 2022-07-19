import copy
from types import SimpleNamespace
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from federatedml.nn.backend.tf_keras.nn_model import KerasNNModel

try:
    import torch
    import torch as t
    from federatedml.nn.backend.fate_torch.nn import Linear
    from federatedml.nn.backend.fate_torch.nn import Sequential as tSequential
except ImportError:
    pass


def modify_linear_input_shape(input_shape, layer_config):
    new_layer_config = copy.deepcopy(layer_config)
    for k, v in new_layer_config.items():
        if v['layer'] == 'Linear':
            v['in_features'] = input_shape
    return new_layer_config


def torch_interactive_to_keras_nn_model(seq):

    linear_layer = seq[0]  # take the linear layer
    weight = linear_layer.weight.detach().numpy()
    if linear_layer.bias is not None:
        bias = linear_layer.bias.detach().numpy()
    else:
        bias = None

    in_shape = weight.shape[1]
    out_shape = weight.shape[0]
    weight = weight.transpose()
    use_bias = not (bias is None)
    keras_model = KerasNNModel(Sequential(Dense(units=out_shape, input_shape=(in_shape, ), use_bias=use_bias)))
    if bias is not None:
        keras_model._model.layers[0].set_weights([weight, bias])
    else:
        keras_model._model.layers[0].set_weights([weight])

    keras_model.compile('keep_predict_loss', optimizer=SimpleNamespace(optimizer="SGD", kwargs={}), metrics=None)

    return keras_model


def keras_nn_model_to_torch_linear(keras_model: KerasNNModel):

    weights = keras_model.get_trainable_weights()
    use_bias = len(weights) == 2
    in_shape, out_shape = weights[0].shape[0], weights[0].shape[1]
    linear = Linear(in_shape, out_shape, use_bias)

    with torch.no_grad():
        linear.weight = torch.nn.Parameter(t.Tensor(weights[0].transpose()))
        if use_bias:
            linear.bias = torch.nn.Parameter(t.Tensor(weights[1]))

    return tSequential(linear)


def pytorch_label_reformat(labels):

    if labels.shape[1] == 1:  # binary classification
        return labels
    else:
        return t.Tensor(labels).argmax(dim=1).flatten().numpy()  # multi classification


if __name__ == '__main__':

    import numpy as np

    test_data = np.random.random((3, 10))
    test_data_t = t.Tensor(test_data)
    a = tSequential(t.nn.Linear(10, 5, True))
    model = torch_interactive_to_keras_nn_model(a)
    new_linear = keras_nn_model_to_torch_linear(model)
