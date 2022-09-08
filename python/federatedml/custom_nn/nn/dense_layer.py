import torch as t
import pandas as pd
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from federatedml.custom_nn.nn_base_module import NNBaseModule
from federatedml.util import LOGGER
from federatedml.util import consts


class TestNet(nn.Module):

    def __init__(self, input_size):
        super(TestNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.activation = nn.Sigmoid()

    def forward(self, input_data):
        out = self.seq(input_data)
        return self.activation(out)


class TestDataset(Dataset):

    def __init__(self, feat, label):
        self.feat = feat
        self.label = label

    def __getitem__(self, item):
        return self.feat[item], self.label[item]

    def __len__(self):
        return len(self.label)


class TestNetModule(NNBaseModule):

    def __init__(self):
        super(TestNetModule, self).__init__()
        self.model = None

    def train(self, cpn_input, **kwargs):

        LOGGER.debug('input data is {}'.format(cpn_input))

        epochs = 100
        lr = 0.01
        test_batch_size = 256

        if self.role == consts.GUEST:
            df_train = pd.read_csv('/home/cwj/standalone_fate_install_1.9.0_release/examples/data/epsilon_5k_homo_guest.csv')
        elif self.role == consts.HOST:
            df_train = pd.read_csv('/home/cwj/standalone_fate_install_1.9.0_release/examples/data/epsilon_5k_homo_host.csv')
        else:
            raise ValueError('cannot get correct role info and read training data, role is {}'.format(self.role))

        label = np.array(df_train['y']).astype(np.float32)
        id_ = df_train['id']
        features = df_train.drop(columns=['id', 'y']).values
        features = np.array(features).astype(np.float32)
        dataset = TestDataset(features, label)
        dl = DataLoader(dataset, batch_size=test_batch_size)

        # set training communication round
        self.set_fed_avg_round_num(comm_round=epochs)

        self.model = TestNet(100)
        optimizer = t.optim.Adam(self.model.parameters(), lr=lr)
        loss_func = t.nn.BCELoss()

        for i in range(epochs):
            LOGGER.debug('epoch is {}'.format(i))
            epoch_loss = 0
            batch_idx = 0
            for batch_data, batch_label in dl:
                optimizer.zero_grad()
                pred = self.model(batch_data)
                batch_loss = loss_func(pred.flatten(), batch_label.flatten())
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.detach().numpy()
                batch_idx += 1
            epoch_loss = epoch_loss / batch_idx

            # federation process
            LOGGER.debug('epoch loss is {}'.format(epoch_loss))
            self.fed_avg_model(optimizer, loss=epoch_loss, loss_weight=len(dataset))

        from sklearn.metrics import roc_auc_score
        train_pred = self.model(t.Tensor(features)).detach().numpy()
        LOGGER.debug('final train auc is {}'.format(roc_auc_score(label, train_pred)))

    def predict(self, cpn_input, **kwargs):

        df_train = pd.read_csv('/home/cwj/standalone_fate_install_1.9.0_release/examples/data/breast_homo_guest.csv')
        label = np.array(df_train['y']).astype(np.float32)
        id_ = df_train['id']
        features = df_train.drop(columns=['id', 'y']).values
        features = np.array(features).astype(np.float32)
        features = t.Tensor(features)

        # return self.model(features).detach().numpy()


if __name__ == '__main__':

    model = TestNetModule()
    model.set_role(consts.GUEST)
    model.local_mode()
    model.train(None)
    # pred_rs = model.predict(None)

