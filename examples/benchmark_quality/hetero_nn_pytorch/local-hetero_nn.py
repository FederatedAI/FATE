import argparse
import numpy as np
import os
from tensorflow import keras
import pandas
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from sklearn import metrics
from pipeline.utils.tools import JobConfig
from sklearn.preprocessing import LabelEncoder

import torch as t
from torch import nn
from torch.utils.data import Dataset, DataLoader
import tqdm
from pipeline import fate_torch_hook
fate_torch_hook(t)


class TestModel(t.nn.Module):

    def __init__(self, guest_input_shape, host_input_shape):
        super(TestModel, self).__init__()

        self.guest_bottom = t.nn.Sequential(
            nn.Linear(guest_input_shape, 10, True),
            nn.ReLU(),
            nn.Linear(10, 8, True),
            nn.ReLU()
        )

        self.host_bottom = t.nn.Sequential(
            nn.Linear(host_input_shape, 10, True),
            nn.ReLU(),
            nn.Linear(10, 8, True),
            nn.ReLU()
        )

        self.inter_a, self.inter_b = t.nn.Linear(8, 4, True), t.nn.Linear(8, 4, True)

        self.top_model_guest = t.nn.Sequential(
            nn.Linear(4, 1, True),
            nn.Sigmoid()
        )

    def forward(self, data):
        x_guest, x_host = data[0].type(t.float), data[1].type(t.float)
        guest_fw = self.inter_a(self.guest_bottom(x_guest))
        host_fw = self.inter_b(self.host_bottom(x_host))
        out = self.top_model_guest(guest_fw + host_fw)
        return out

    def predict(self, data):
        rs = self.forward(data)
        return rs.detach().numpy()


class TestDataset(Dataset):

    def __init__(self, guest_data, host_data, label):
        super(TestDataset, self).__init__()
        self.g = guest_data
        self.h = host_data
        self.l = label

    def __getitem__(self, idx):
        return self.g[idx], self.h[idx], self.l[idx]

    def __len__(self):
        return len(self.l)


def build(param, shape1, shape2):
    return TestModel(shape1, shape2)


def main(config="./config.yaml", param="./hetero_nn_breast_config.yaml"):

    try:
        if isinstance(config, str):
            config = JobConfig.load_from_file(config)
            data_base_dir = config["data_base_dir"]
        else:
            data_base_dir = config.data_base_dir
        if isinstance(param, str):
            param = JobConfig.load_from_file(param)
        data_guest = param["data_guest"]
        data_host = param["data_host"]
        idx = param["idx"]
        label_name = param["label_name"]
        # prepare data
        Xb = pandas.read_csv(os.path.join(data_base_dir, data_guest), index_col=idx)
        Xa = pandas.read_csv(os.path.join(data_base_dir, data_host), index_col=idx)
        y = Xb[label_name]
        out = Xa.drop(Xb.index)
        Xa = Xa.drop(out.index)
        Xb = Xb.drop(label_name, axis=1)
        # torch model
        model = build(param, Xb.shape[1], Xa.shape[1])
        Xb = t.Tensor(Xb.values)
        Xa = t.Tensor(Xa.values)
        y = t.Tensor(y.values)
        dataset = TestDataset(Xb, Xa, y)
        batch_size = len(dataset) if param['batch_size'] == -1 else param['batch_size']
        dataloader = DataLoader(dataset, batch_size=batch_size)
        optimizer = t.optim.Adam(lr=param['learning_rate']).to_torch_instance(model.parameters())

        if param['eval_type'] == 'binary':
            loss_fn = t.nn.BCELoss()

        for i in tqdm.tqdm(range(param['epochs'])):

            for gd, hd, label in dataloader:
                optimizer.zero_grad()
                pred = model([gd, hd])
                loss = loss_fn(pred.flatten(), label.type(t.float32))
                loss.backward()
                optimizer.step()

        eval_result = {}
        for metric in param["metrics"]:
            if metric.lower() == "auc":
                predict_y = model.predict([Xb, Xa])
                auc = metrics.roc_auc_score(y, predict_y)
                eval_result["auc"] = auc
            elif metric == "accuracy":
                predict_y = np.argmax(model.predict([Xb, Xa]), axis=1)
                predict_y = label_encoder.inverse_transform(predict_y)
                acc = metrics.accuracy_score(y_true=labels, y_pred=predict_y)
                eval_result["accuracy"] = acc

        data_summary = {}
    except Exception as e:
        print(e)
    return data_summary, eval_result


if __name__ == "__main__":

    parser = argparse.ArgumentParser("BENCHMARK-QUALITY SKLEARN JOB")
    parser.add_argument("-config", type=str,
                        help="config file")
    parser.add_argument("-param", type=str,
                        help="config file for params")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config, args.param)
    else:
        main()
