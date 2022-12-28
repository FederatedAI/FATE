import argparse
import numpy as np
import os
import pandas

from sklearn import metrics
from pipeline.utils.tools import JobConfig

import torch as t
from torch import nn
from pipeline import fate_torch_hook
from torch.utils.data import DataLoader, TensorDataset
from federatedml.nn.backend.utils.common import global_seed
fate_torch_hook(t)


class HeteroLocalModel(t.nn.Module):

    def __init__(self, guest_btn, host_btn, interactive, top):
        super().__init__()
        self.guest_btn = guest_btn
        self.host_btn = host_btn
        self.inter = interactive
        self.top = top

    def forward(self, x1, x2):
        return self.top(self.inter(self.guest_btn(x1), self.host_btn(x2)))


def build(param, shape1, shape2, lr):

    global_seed(101)
    guest_bottom = t.nn.Sequential(
        nn.Linear(shape1, param["bottom_layer_units"]),
        nn.ReLU()
    )

    host_bottom = t.nn.Sequential(
        nn.Linear(shape2, param["bottom_layer_units"]),
        nn.ReLU()
    )

    interactive_layer = t.nn.InteractiveLayer(
        guest_dim=param["bottom_layer_units"],
        host_dim=param["bottom_layer_units"],
        host_num=1,
        out_dim=param["interactive_layer_units"])

    act = nn.Sigmoid() if param["top_layer_units"] == 1 else nn.Softmax(dim=1)
    top_layer = t.nn.Sequential(
        t.nn.Linear(
            param["interactive_layer_units"],
            param["top_layer_units"]),
        act)

    model = HeteroLocalModel(
        guest_bottom,
        host_bottom,
        interactive_layer,
        top_layer)
    opt = t.optim.Adam(model.parameters(), lr=lr)
    return model, opt


def fit(epoch, model, optimizer, loss, batch_size, dataset):

    print(
        'model is {}, loss is {}, optimizer is {}'.format(
            model,
            loss,
            optimizer))
    dl = DataLoader(dataset, batch_size=batch_size)
    for i in range(epoch):
        epoch_loss = 0
        for xa, xb, label in dl:
            optimizer.zero_grad()
            pred = model(xa, xb)
            l = loss(pred, label)
            epoch_loss += l.detach().numpy()
            l.backward()
            optimizer.step()
        print('epoch is {}, epoch loss is {}'.format(i, epoch_loss))


def predict(model, Xa, Xb):

    pred_rs = model(Xb, Xa)
    return pred_rs.detach().numpy()


def main(config="../../config.yaml", param="./hetero_nn_breast_config.yaml"):

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
    Xb = pandas.read_csv(
        os.path.join(
            data_base_dir,
            data_guest),
        index_col=idx)
    Xa = pandas.read_csv(os.path.join(data_base_dir, data_host), index_col=idx)
    y = Xb[label_name]
    out = Xa.drop(Xb.index)
    Xa = Xa.drop(out.index)
    Xb = Xb.drop(label_name, axis=1)

    Xa = t.Tensor(Xa.values)
    Xb = t.Tensor(Xb.values)
    y = t.Tensor(y.values)

    if param["loss"] == "categorical_crossentropy":
        loss = t.nn.CrossEntropyLoss()
        y = y.type(t.int64).flatten()
    else:
        loss = t.nn.BCELoss()
        y = y.reshape((-1, 1))

    model, opt = build(
        param, Xb.shape[1], Xa.shape[1], lr=param['learning_rate'])

    dataset = TensorDataset(Xb, Xa, y)
    fit(epoch=param['epochs'], model=model, optimizer=opt,
        batch_size=param['batch_size'], dataset=dataset, loss=loss)

    eval_result = {}
    for metric in param["metrics"]:
        if metric.lower() == "auc":
            predict_y = predict(model, Xa, Xb)
            auc = metrics.roc_auc_score(y, predict_y)
            eval_result["auc"] = auc
        elif metric == "accuracy":
            predict_y = np.argmax(predict(model, Xa, Xb), axis=1)
            acc = metrics.accuracy_score(
                y_true=y.detach().numpy(), y_pred=predict_y)
            eval_result["accuracy"] = acc

    print(eval_result)
    data_summary = {}
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
    main()
