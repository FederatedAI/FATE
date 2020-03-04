import copy
import io
import os
import tempfile
import zipfile
import collections
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
from arch.api.utils import log_utils
from federatedml.framework.weights import OrderDictWeights, Weights
from federatedml.nn.homo_nn.nn_model import NNModel, DataConverter

Logger = log_utils.getLogger()


def layers(layer, config, type):
    if type == "cv":
       if layer == "Conv2d":
           return torch.nn.Conv2d()
       if layer == "MaxPool2d":
           return torch.nn.MaxPool2d()
       if layer == "AvgPool2d":
           return torch.nn.AvgPool2d()
    elif type == "nlp":
        if layer == "LSTM":
            return torch.nn.LSTM()
        if layer == "RNN":
            return torch.nn.RNN()
    elif type == "activate":
        if layer == "Sigmoid":
            return torch.nn.Sigmoid()
        if layer == "Relu":
            return torch.nn.ReLU()
        if layer == "Selu":
            return torch.nn.SELU()
        if layer == "LeakyReLU":
            return torch.nn.LeakyReLU()
        if layer == "Tanh":
            return torch.nn.Tanh()

    else:
        if layer == "Linear":
            return torch.nn.Linear(config[0], config[1])
        if layer == "BatchNorm2d":
            return torch.nn.BatchNorm2d()
        if layer == "dropout":
            return torch.nn.Dropout()

def build_pytorch(nn_define, optimizer, loss):
    model = torch.nn.Sequential()
    for config in nn_define:
        layer = layers(config.get("layer"), config.get("config"), config.get("type"))
        model.add_module(config.get("name"), layer)
    return PytorchNNModel(model, optimizer, loss)


def build_loss_fn(loss):
    if loss == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss()
    elif loss == "MSELoss":
        return torch.nn.MSELoss()
    elif loss == "BCELoss":
        return torch.nn.BCELoss()
    elif loss == "BCEWithLogitsLoss":
        return torch.nn.BCEWithLogitsLoss()
    elif loss == "NLLLoss":
        return torch.nn.NLLLoss()
    elif loss == "L1Loss":
        return torch.nn.L1Loss()
    elif loss == "SmoothL1Loss":
        return torch.nn.SmoothL1Loss()
    else:
        print("loss function not support!")

def build_optimzer(optim, model):
    if optim.optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), lr=optim.kwargs.get("learning_rate"))
    elif optim.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), lr=optim.kwargs.get("learning_rate"))
    elif optim.optimizer == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), lr=optim.kwargs.get("learning_rate"))
    elif optim.optimizer == "Adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=optim.kwargs.get("learning_rate"))
    else:
        print("not support")


def restore_pytorch_nn_model(model_bytes):
    return PytorchNNModel.restore_model(model_bytes)


class PytorchNNModel(NNModel):

    def __init__(self, model, optimizer=None, loss=None):
        self._model: torch.nn.Sequential = model
        self._optimizer = optimizer
        self._loss = loss

    def get_model_weights(self) -> OrderDictWeights:
        return OrderDictWeights(self._model.state_dict())

    def set_model_weights(self, weights: Weights):
        unboxed = weights.unboxed
        self._model.load_state_dict(unboxed)

    def train(self, data: data.Dataset, **kwargs):
        loss_fn = build_loss_fn(self._loss)
        optimizer = build_optimzer(self._optimizer, self._model)
        epochs = 1
        left_kwargs = copy.deepcopy(kwargs)
        if "aggregate_every_n_epoch" in kwargs:
            epochs = kwargs["aggregate_every_n_epoch"]
            del left_kwargs["aggregate_every_n_epoch"]
        train_data = DataLoader(data, batch_size=data.batch_size, shuffle=False)
        for epoch in range(epochs):
            for batch_id, (feature, label) in enumerate(train_data):
                feature = torch.tensor(feature, dtype=torch.float32)
                label = torch.tensor(label, dtype=torch.float32)
                y_pre = self._model(feature)
                optimizer.zero_grad()
                loss = loss_fn(y_pre, label)
                loss.backward()
                optimizer.step()


    def evaluate(self, data: data.dataset, **kwargs):
        metircs={}
        metircs["loss"]=0
        metircs["auccuray"]=0
<<<<<<< HEAD
        loss_fn = build_loss_fn(self._loss)
=======
>>>>>>> e85522637330fc9cd8a252192c14e817e5f48849
        self._model.eval()
        evaluate_data = DataLoader(data, batch_size=1, shuffle=False)
        result = np.zeros((len(data), data.y_shape[0]))
        eval_label = np.zeros((len(data), data.y_shape[0]))
        num_output_units = data.get_shape()[1]
        index = 0
        eval_loss = 0
        for batch_id, (feature, label) in enumerate(evaluate_data):
            feature = torch.tensor(feature, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)
            y = self._model(feature)
            loss = loss_fn(y, label)
            eval_loss += loss
            result[index] = y.detach().numpy()
            eval_label[index] = label.detach().numpy()
            index += 1
        metircs["loss"] = eval_loss.item()
        acc=0
        if num_output_units[0] == 1:
            for i in range(len(data)):
                if(result[i]>0.5):
                   if(eval_label[i]==1):
                       acc+=1;
                else:
                    if(eval_label[i]==0):
                       acc+=1;
        else:
            for i in range(len(data)):
                if(result[i].argmax()==eval_label[i].argmax()):
                   acc+=1;

        metircs["auccuray"] = acc / len(data)
        return metircs

    def predict(self, data: data.dataset, **kwargs):

        result = np.zeros((len(data), data.y_shape[0]))
        predict_data = DataLoader(data, batch_size=1, shuffle=False)
        index = 0
        for batch_id, (feature, label) in enumerate(predict_data):
            feature = torch.tensor(feature, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.float32)
            y = self._model(feature)
            result[index] = y.detach().numpy()
            index += 1
        return result

    def export_model(self):
        f = tempfile.TemporaryFile()
        try:
            torch.save(self._model, f)
            f.seek(0)
            model_bytes = f.read()
            return model_bytes
        finally:
            f.close()

    def restore_model(model_bytes):
        f = tempfile.TemporaryFile()
        f.write(model_bytes)
        f.seek(0)
        model = torch.load(f)
        f.close()
        return PytorchNNModel(model)

# class PredictNN(NNModel):
#     def __init__(self, model):
#         self._model: torch.nn.Sequential = model
#
#     def predict(self, data: data.dataset, **kwargs):
#         # size = len(data)
#         result = np.zeros((len(data), data.y_shape[0]))
#         predict_data = DataLoader(data, batch_size=1, shuffle=False)
#         index = 0
#         for batch_id, (feature, label) in enumerate(predict_data):
#             feature = torch.tensor(feature, dtype=torch.float32)
#             label = torch.tensor(label, dtype=torch.float32)
#             y = self._model(feature)
#             result[index] = y.detach().numpy()
#             index += 1
#         return result
#
#     def export_model(self):
#         f = tempfile.TemporaryFile()
#         try:
#             torch.save(self._model, f)
#             f.seek(0)
#             model_bytes = f.read()
#             return model_bytes
#         finally:
#             f.close()
#
#     def restore_model(model_bytes):
#         f = tempfile.TemporaryFile()
#         f.write(model_bytes)
#         f.seek(0)
#         model = torch.load(f)
#         f.close()
#         return PredictNN(model)
#

class PytorchData(data.Dataset):
    def __init__(self, data_instances, batch_size):
        self.size = data_instances.count()

        if self.size <= 0:
            raise ValueError("empty data")

        if batch_size == -1:
            self.batch_size = self.size
        else:
            self.batch_size = batch_size
        _, one_data = data_instances.first()
        self.x_shape = one_data.features.shape

        num_label = len(data_instances.map(lambda x, y: [x, {y.label}]).reduce(lambda x, y: x | y))
        if num_label == 2:
            self.y_shape = (1,)
            self.x = np.zeros((self.size, *self.x_shape))
            self.y = np.zeros((self.size, *self.y_shape))
            index = 0
            self._keys = []
            for k, inst in data_instances.collect():
                self._keys.append(k)
                self.x[index] = inst.features
                self.y[index] = inst.label
                index += 1

        # encoding label in one-hot
        elif num_label > 2:
            self.y_shape = (num_label,)
            self.x = np.zeros((self.size, *self.x_shape))
            self.y = np.zeros((self.size, *self.y_shape))
            index = 0
            self._keys = []
            for k, inst in data_instances.collect():
                self._keys.append(k)
                self.x[index] = inst.features
                self.y[index][inst.label] = 1
                index += 1
        else:
            raise ValueError(f"num_label is {num_label}")

    def __getitem__(self, index):

        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def get_keys(self):
        return self._keys

    def get_shape(self):
        return self.x_shape, self.y_shape


class PytorchDataConverter(DataConverter):
    def convert(self, data, *args, **kwargs):
        return PytorchData(data, *args, **kwargs)

