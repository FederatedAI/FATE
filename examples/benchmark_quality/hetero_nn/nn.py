import argparse
import os
import torch as t
import pandas as pd
import math
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score, roc_curve, mean_absolute_error, mean_squared_error
from fate_client.pipeline.utils.test_utils import JobConfig


class FakeNNModel(t.nn.Module):

    def __init__(self, guest_bottom, host_bottom, guest_top, agg_layer_guest, agg_layer_host, is_binary=True):
        super(FakeNNModel, self).__init__()
        self.guest_bottom = t.nn.Linear(guest_bottom[0], guest_bottom[1])
        self.host_bottom = t.nn.Linear(host_bottom[0], host_bottom[1])
        self.guest_top = t.nn.Linear(guest_top[0], guest_top[1])
        self.agg_layer_guest = t.nn.Linear(agg_layer_guest[0], agg_layer_guest[1])
        self.agg_layer_host = t.nn.Linear(agg_layer_host[0], agg_layer_host[1])
        self.is_binary = is_binary

    def forward(self, x_g, x_h):
        x_g = self.guest_bottom(x_g)
        x_h = self.host_bottom(x_h)
        x_g = self.agg_layer_guest(x_g)
        x_h = self.agg_layer_host(x_h)
        x = x_g + x_h
        x = self.guest_top(x)
        if self.is_binary:
            x = t.nn.Sigmoid()(x)
        return x
    


def main(config="../../config.yaml", param="./default_credit_config.yaml"):

    if isinstance(param, str):
        param = JobConfig.load_from_file(param)
        print('param is {}'.format(param))

    if isinstance(config, str):
        config = JobConfig.load_from_file(config)
        print(f"config: {config}")
        data_base_dir = config["data_base_dir"]
    else:
        data_base_dir = config.data_base_dir

    print(data_base_dir)
    data_dir_path = str(data_base_dir) + '/'

    model = FakeNNModel(param['guest_model']['bottom'], param['host_model']['bottom'], param['guest_model']['top'],
                        param['guest_model']['agg_layer'], param['host_model']['agg_layer'], param['is_binary'])
    guest_data_path = param['data_guest']
    host_data_path = param['data_host']
    id_name = param['id_name']
    label_name = param['label_name']

    df_g = pd.read_csv(data_dir_path + guest_data_path)
    df_g = df_g.drop(columns=[id_name])
    label = df_g[label_name].values.reshape(-1, 1)
    df_g = df_g.drop(columns=[label_name])
    df_h = pd.read_csv(data_dir_path + host_data_path)
    df_h = df_h.drop(columns=[id_name])

    # tensor dataset
    dataset = t.utils.data.TensorDataset(t.tensor(df_g.values).float(), t.tensor(df_h.values).float(),
                                         t.tensor(label).float())

    epoch = param['epochs']
    batch_size = param['batch_size']
    lr = param['lr']

    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    data_loader = t.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for i in range(epoch):

        loss_sum = 0
        for x_g, x_h, label in data_loader:
            optimizer.zero_grad()
            output = model(x_g, x_h)
            loss = t.nn.BCELoss()(output, label)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        print(f'epoch {i} loss: {loss_sum}')

    # predict
    y_prob = []
    true_label = []
    for x_g, x_h, label in data_loader:
        output = model(x_g, x_h)
        y_prob.append(output.detach().numpy())
        true_label.append(label.detach().numpy())

    if param['is_binary']:
        # compute auc
        import numpy as np
        from sklearn.metrics import roc_auc_score
        y_prob = np.concatenate(y_prob)
        y_prob = y_prob.reshape(-1)
        y_true = np.concatenate(true_label)
        auc_score = roc_auc_score(y_true, y_prob)
        print(f'auc score: {auc_score}')
        return {}, {'auc': auc_score}
    else:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser("BENCHMARK-QUALITY XGBoost JOB")
    parser.add_argument("-c", "--config", type=str,
                        help="config file", default="../../config.yaml")
    parser.add_argument("-p", "--param", type=str,
                        help="config file for params", default="./breast_config.yaml")
    args = parser.parse_args()
    main(args.config, args.param)