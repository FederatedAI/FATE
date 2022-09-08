import torch as t
import pandas as pd
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from federatedml.custom_nn.nn_base_module import NNBaseModule
from federatedml.util import LOGGER
from federatedml.util import consts


class MF(nn.Module):

    def __init__(self, user_num, item_num, embedding_size):
        super(MF, self).__init__()
        self.user_embedding = nn.Embedding(embedding_dim=embedding_size, num_embeddings=user_num)
        self.item_embedding = nn.Embedding(embedding_dim=embedding_size, num_embeddings=item_num)
        self.sigmoid = nn.Sigmoid()

    def forward(self, u, i):
        u_embed = self.user_embedding(u)
        i_embed = self.item_embedding(i)
        pred = (u_embed * i_embed).sum(axis=1)
        return self.sigmoid(pred)


class TestDataset(Dataset):

    def __init__(self, user_id, item_id, rating):
        self.u = user_id
        self.i = item_id
        self.r = rating

    def __getitem__(self, item):
        return self.u[item], self.i[item], self.r[item]

    def __len__(self):
        return len(self.r)


class MFRecModule(NNBaseModule):

    def __init__(self):
        super(MFRecModule, self).__init__()
        self.model = None

    def train(self, cpn_input, **kwargs):

        LOGGER.debug('input data is {}'.format(cpn_input))

        epochs = kwargs['epochs']
        lr = kwargs['lr']
        test_batch_size = kwargs['batch_size']

        if self.role == consts.GUEST:
            df_train = pd.read_csv('/home/cwj/standalone_fate_install_1.9.0_release/examples/data/anime_homo_guest.csv')
        elif self.role == consts.HOST:
            df_train = pd.read_csv('/home/cwj/standalone_fate_install_1.9.0_release/examples/data/anime_homo_host.csv')
        else:
            raise ValueError('cannot get correct role info and read training data, role is {}'.format(self.role))

        label = np.array(df_train['rating']).astype(np.float32)
        uid = df_train['user_id'].astype(np.int64)
        iid = df_train['anime_id'].astype(np.int64)
        dataset = TestDataset(user_id=uid, item_id=iid, rating=label)
        dl = DataLoader(dataset, batch_size=test_batch_size)

        # set training communication round
        self.set_fed_avg_round_num(comm_round=epochs)

        self.model = MF(user_num=uid.max()+1, item_num=iid.max()+1, embedding_size=32)
        optimizer = t.optim.Adam(self.model.parameters(), lr=lr)
        loss_func = t.nn.BCELoss()

        import tqdm
        for i in range(epochs):
            LOGGER.debug('epoch is {}'.format(i))
            epoch_loss = 0
            batch_idx = 0
            for uid_batch, iid_batch, r_batch in tqdm.tqdm(dl):
                optimizer.zero_grad()
                pred = self.model(uid_batch, iid_batch)
                batch_loss = loss_func(pred.flatten(), r_batch.flatten())
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.detach().numpy()
                batch_idx += 1
            epoch_loss = epoch_loss / batch_idx

            # federation process
            LOGGER.debug('epoch loss is {}'.format(epoch_loss))
            self.fed_avg_model(optimizer, loss=epoch_loss, loss_weight=len(dataset))

        # from sklearn.metrics import roc_auc_score
        # train_pred = self.model(t.Tensor(features)).detach().numpy()
        # LOGGER.debug('final train auc is {}'.format(roc_auc_score(label, train_pred)))

    def predict(self, cpn_input, **kwargs):
        pass


if __name__ == '__main__':

    model = MFRecModule()
    model.set_role(consts.GUEST)
    model.local_mode()
    model.train(None)
    # pred_rs = model.predict(None)
