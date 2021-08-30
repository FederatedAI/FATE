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
import copy
import tempfile

import numpy as np
import torch
import torch.utils.data as data
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    roc_auc_score,
    recall_score,
    f1_score,
    fbeta_score,
)
from torch.utils.data import DataLoader

from federatedml.framework.weights import OrderDictWeights, Weights
from federatedml.nn.homo_nn.nn_model import NNModel, DataConverter


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
        if layer == "Softmax":
            return torch.nn.Softmax(1)
        if layer == "LogSoftmax":
            return torch.nn.LogSoftmax(1)

    elif type == "normal":
        if layer == "Linear":
            return torch.nn.Linear(config[0], config[1])
        if layer == "BatchNorm2d":
            return torch.nn.BatchNorm2d()
        if layer == "dropout":
            return torch.nn.Dropout(config)

    else:
        print("layer not support!")


def build_pytorch(nn_define, optimizer, loss, metrics, **kwargs):
    model = torch.nn.Sequential()
    for config in nn_define:
        layer = layers(config.get("layer"), config.get("config"), config.get("type"))
        model.add_module(config.get("name"), layer)
    return PytorchNNModel(model, optimizer, loss, metrics)


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
    elif loss == "HingeEmbeddingLoss":
        return torch.nn.HingeEmbeddingLoss()
    else:
        print("loss function not support!")


def build_optimzer(optim, model):
    if optim.optimizer == "Adam":
        return torch.optim.Adam(model.parameters(), **optim.kwargs)
    elif optim.optimizer == "SGD":
        return torch.optim.SGD(model.parameters(), **optim.kwargs)
    elif optim.optimizer == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), **optim.kwargs)
    elif optim.optimizer == "Adagrad":
        return torch.optim.Adagrad(model.parameters(), **optim.kwargs)
    else:
        print("not support")


class PytorchNNModel(NNModel):
    def __init__(self, model, optimizer=None, loss=None, metrics=None):
        self._model: torch.nn.Sequential = model
        self._optimizer = optimizer
        self._loss = loss
        self._metrics = metrics

    def compile(self, loss, optimizer, metrics):
        self._optimizer = optimizer
        self._loss = loss
        self._metrics = metrics

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
                if isinstance(loss_fn, torch.nn.CrossEntropyLoss) | isinstance(
                    loss_fn, torch.nn.NLLLoss
                ):
                    label = torch.tensor(label, dtype=torch.long)
                    temp = label.t()
                    label = temp[0]
                else:
                    label = torch.tensor(label, dtype=torch.float32)
                y_pre = self._model(feature)
                optimizer.zero_grad()
                loss = loss_fn(y_pre, label)
                loss.backward()
                optimizer.step()

    def evaluate(self, data: data.dataset, **kwargs):

        metircs = {}
        loss_metircs = []
        loss_fuc = []
        other_metrics = []
        if self._metrics:
            for func in self._metrics:
                if func.endswith("Loss"):
                    loss_metircs.append(func)
                    loss_fuc.append(build_loss_fn(func))
                else:
                    other_metrics.append(func)
        self._model.eval()
        loss_fn = build_loss_fn(self._loss)
        evaluate_data = DataLoader(data, batch_size=data.batch_size, shuffle=False)
        if loss_metircs:
            loss_metircs_result = [0 for i in range(len(loss_metircs))]

        loss = 0
        for batch_id, (feature, label) in enumerate(evaluate_data):
            feature = torch.tensor(feature, dtype=torch.float32)
            y = self._model(feature)
            if batch_id == 0:
                result = y.detach().numpy()
                eval_label = label.detach().numpy()
            else:
                result = np.vstack((result, y.detach().numpy()))
                eval_label = np.vstack((eval_label, label.detach().numpy()))
            if isinstance(loss_fn, torch.nn.CrossEntropyLoss) | isinstance(
                loss_fn, torch.nn.NLLLoss
            ):
                label = torch.tensor(label, dtype=torch.long)
                temp = label.t()
                label = temp[0]
            else:
                label = torch.tensor(label, dtype=torch.float32)
            eval_loss = loss_fn(y, label)
            if loss_metircs:
                for i in range(len(loss_fuc)):
                    f = loss_fuc[i]
                    res = f(y, label)
                    loss_metircs_result[i] += res.item()
            loss += eval_loss

        metircs["loss"] = loss.item() * data.batch_size / len(data)
        if loss_metircs:
            i = 0
            for func in loss_metircs:
                metircs[func] = loss_metircs_result[i] * data.batch_size / len(data)
                i += 1
        num_output_units = result.shape[1]
        if len(other_metrics) > 0:
            if num_output_units == 1:
                for i in range(len(data)):
                    if result[i] > 0.5:
                        result[i] = 1
                    else:
                        result[i] = 0
                for fuc_name in other_metrics:
                    if fuc_name == "auccuray":
                        metircs[str(fuc_name)] = accuracy_score(result, eval_label)
                    elif fuc_name == "precision":
                        metircs[str(fuc_name)] = precision_score(result, eval_label)
                    elif fuc_name == "recall":
                        metircs[str(fuc_name)] = recall_score(result, eval_label)
                    elif fuc_name == "auc":
                        metircs[str(fuc_name)] = roc_auc_score(result, eval_label)
                    elif fuc_name == "f1":
                        metircs[str(fuc_name)] = f1_score(result, eval_label)
                    elif fuc_name == "fbeta":
                        metircs[str(fuc_name)] = fbeta_score(result, eval_label, beta=2)
                    else:
                        print("metrics not support ")
            else:
                acc = 0
                if data.use_one_hot:
                    for i in range(len(data)):
                        if result[i].argmax() == eval_label[i].argmax():
                            acc += 1
                else:
                    for i in range(len(data)):
                        if result[i].argmax() == eval_label[i]:
                            acc += 1
                metircs["auccuray"] = acc / len(data)
        return metircs

    def predict(self, data: data.dataset, **kwargs):

        predict_data = DataLoader(data, batch_size=data.batch_size, shuffle=False)
        for batch_id, (feature, label) in enumerate(predict_data):
            feature = torch.tensor(feature, dtype=torch.float32)
            # label = torch.tensor(label, dtype=torch.float32)
            y = self._model(feature)
            if batch_id == 0:
                result = y.detach().numpy()
            else:
                result = np.vstack((result, y.detach().numpy()))
        return result

    def export_model(self):
        with tempfile.TemporaryFile() as f:
            torch.save(self._model, f)
            f.seek(0)
            return f.read()

    @classmethod
    def restore_model(cls, model_bytes):
        with tempfile.TemporaryFile() as f:
            f.write(model_bytes)
            f.seek(0)
            model = torch.load(f)
        return PytorchNNModel(model)


class PytorchData(data.Dataset):
    def __init__(self, data_instances, batch_size, encode_label, label_mapping):
        self.size = data_instances.count()
        if self.size <= 0:
            raise ValueError("empty data")
        if batch_size == -1:
            self.batch_size = self.size
        else:
            self.batch_size = batch_size
        _, one_data = data_instances.first()
        self.x_shape = one_data.features.shape

        num_label = len(label_mapping)
        # encoding label in one-hot
        if encode_label:
            self.use_one_hot = True
            if num_label > 2:
                self.y_shape = (num_label,)
                self.output_shape = self.y_shape
                self.x = np.zeros((self.size, *self.x_shape))
                self.y = np.zeros((self.size, *self.y_shape))
                index = 0
                self._keys = []
                for k, inst in data_instances.collect():
                    self._keys.append(k)
                    self.x[index] = inst.features
                    self.y[index][label_mapping[inst.label]] = 1
                    index += 1
            else:
                raise ValueError(f"num_label is {num_label}")
        else:
            self.use_one_hot = False
            self.y_shape = (1,)
            if num_label == 2:
                self.output_shape = self.y_shape
            elif num_label > 2:
                self.output_shape = (num_label,)
            else:
                raise ValueError(f"num_label is {num_label}")
            self.x = np.zeros((self.size, *self.x_shape))
            self.y = np.zeros((self.size, *self.y_shape))
            index = 0
            self._keys = []
            for k, inst in data_instances.collect():
                self._keys.append(k)
                self.x[index] = inst.features
                self.y[index] = label_mapping[inst.label]
                index += 1

    def __getitem__(self, index):

        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

    def get_keys(self):
        return self._keys

    def get_shape(self):
        return self.x_shape, self.output_shape


class PytorchDataConverter(DataConverter):
    def convert(self, data, *args, **kwargs):
        return PytorchData(data, *args, **kwargs)
