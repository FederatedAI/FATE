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

import argparse
import pathlib
import numpy as np
import torch as t
from torch.utils.data import DataLoader, TensorDataset
import pandas
from pipeline.utils.tools import JobConfig
from federatedml.nn.backend.utils.common import global_seed


dataset = {
    "vehicle": {
        "guest": "examples/data/vehicle_scale_homo_guest.csv",
        "host": "examples/data/vehicle_scale_homo_host.csv",
    },
    "breast": {
        "guest": "examples/data/breast_homo_guest.csv",
        "host": "examples/data/breast_homo_host.csv",
    },
}


def fit(epoch, model, optimizer, loss, batch_size, dataset):

    print(
        'model is {}, loss is {}, optimizer is {}'.format(
            model,
            loss,
            optimizer))
    dl = DataLoader(dataset, batch_size=batch_size)
    for i in range(epoch):
        epoch_loss = 0
        for feat, label in dl:
            optimizer.zero_grad()
            pred = model(feat)
            l = loss(pred, label)
            epoch_loss += l.detach().numpy()
            l.backward()
            optimizer.step()
        print('epoch is {}, epoch loss is {}'.format(i, epoch_loss))


def compute_acc(pred, label, is_multy):

    if is_multy:
        pred = pred.argmax(axis=1)
    else:
        pred = (pred > 0.5) + 0

    return float((pred == label).sum() / len(label))


def main(config="../../config.yaml", param="param_conf.yaml"):

    if isinstance(param, str):
        param = JobConfig.load_from_file(param)
    if isinstance(config, str):
        config = JobConfig.load_from_file(config)
        data_base_dir = config["data_base_dir"]
    else:
        data_base_dir = config.data_base_dir

    epoch = param["epoch"]
    lr = param["lr"]
    batch_size = param.get("batch_size", -1)
    is_multy = param["is_multy"]
    data = dataset[param.get("dataset", "vehicle")]

    global_seed(123)

    if is_multy:
        loss = t.nn.CrossEntropyLoss()
    else:
        loss = t.nn.BCELoss()
    data_path = pathlib.Path(data_base_dir)
    data_with_label = pandas.concat(
        [
            pandas.read_csv(data_path.joinpath(data["guest"]), index_col=0),
            pandas.read_csv(data_path.joinpath(data["host"]), index_col=0),
        ]
    ).values

    data = t.Tensor(data_with_label[:, 1:])
    labels = t.Tensor(data_with_label[:, 0])
    if is_multy:
        labels = labels.type(t.int64)
    else:
        labels = labels.reshape((-1, 1))
    ds = TensorDataset(data, labels)

    input_shape = data.shape[1]
    output_shape = 4 if is_multy else 1
    out_act = t.nn.Softmax(dim=1) if is_multy else t.nn.Sigmoid()

    model = t.nn.Sequential(
        t.nn.Linear(input_shape, 16),
        t.nn.ReLU(),
        t.nn.Linear(16, output_shape),
        out_act
    )

    if batch_size < 0:
        batch_size = len(data_with_label)

    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    fit(epoch, model, optimizer, loss, batch_size, ds)

    pred_rs = model(data)
    acc = compute_acc(pred_rs, labels, is_multy)
    metric_summary = {"accuracy": acc}
    print(metric_summary)
    data_summary = {}
    return data_summary, metric_summary
