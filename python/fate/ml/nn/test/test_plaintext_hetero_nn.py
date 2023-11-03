from fate.ml.nn.model_zoo.hetero_nn_model import HeteroNNModelGuest, HeteroNNModelHost
from fate.ml.nn.model_zoo.agg_layer.agg_layer import AggLayerGuest, AggLayerHost
import sys
from datetime import datetime
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import tqdm


def get_current_datetime_str():
    return datetime.now().strftime("%Y-%m-%d-%H-%M")


guest = ("guest", "10000")
host = ("host", "9999")
name = get_current_datetime_str()


def create_ctx(local, context_name):
    from fate.arch import Context
    from fate.arch.computing.standalone import CSession
    from fate.arch.federation.standalone import StandaloneFederation
    import logging

    # prepare log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # init fate context
    computing = CSession()
    return Context(
        computing=computing, federation=StandaloneFederation(computing, context_name, local, [guest, host])
    )


if __name__ == "__main__":

    party = sys.argv[1]
    import torch as t


    def set_seed(seed):
        t.manual_seed(seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(seed)
            t.backends.cudnn.deterministic = True
            t.backends.cudnn.benchmark = False


    class HeteroNNLocalModel(t.nn.Module):

        def __init__(self, guest_b, guest_t, host_b, merge_type='concat'):
            super(HeteroNNLocalModel, self).__init__()
            self._guest_b = guest_b
            self._guest_t = guest_t
            self._host_b = host_b
            self._merge_type = merge_type

        def forward(self, x_g, x_h):
            fw_g = self._guest_b(x_g)
            fw_h = self._host_b(x_h)
            if self._merge_type == 'concat':
                fw_ = t.cat([fw_g, fw_h], dim=1)
            else:
                fw_ = fw_g + fw_h
            fw_ = self._guest_t(fw_)
            return fw_

    set_seed(42)

    batch_size = 64
    epoch = 5
    guest_bottom = t.nn.Linear(10, 10).double()
    guest_top = t.nn.Sequential(
                                 t.nn.Linear(18, 1),
                                 t.nn.Sigmoid()
                               ).double()
    host_bottom = t.nn.Linear(20, 8).double()

    # # make random fake data
    sample_num = 569

    if party == 'local':

        ctx = create_ctx(guest, get_current_datetime_str())

        df = pd.read_csv('/home/cwj/FATE/FATE-2.0/FATE/examples/data/breast_hetero_guest.csv')
        X_g = t.Tensor(df.drop(columns=['id', 'y']).values).type(t.float64)[0: sample_num]
        y = t.Tensor(df['y'].values).type(t.float64)[0: sample_num]
        df = pd.read_csv('/home/cwj/FATE/FATE-2.0/FATE/examples/data/breast_hetero_host.csv')
        X_h = t.Tensor(df.drop(columns=['id']).values).type(t.float64)[0: sample_num]

        local_model = HeteroNNLocalModel(guest_bottom, guest_top, host_bottom)
        loss_fn = t.nn.BCELoss()
        optimizer = t.optim.Adam(local_model.parameters(), lr=0.01)
        dataset = TensorDataset(X_g, X_h, y)

        for i in range(epoch):
            loss_sum = 0
            batch_idx = 0
            for x_g, x_h, y_ in tqdm.tqdm(DataLoader(dataset, batch_size=batch_size)):
                optimizer.zero_grad()
                fw = local_model(x_g, x_h)
                loss_ = loss_fn(fw.flatten(), y_)
                loss_.backward()
                loss_sum += loss_.item()
                batch_idx += 1
                optimizer.step()
            print(loss_sum / batch_idx)

        pred = local_model(X_g, X_h)
        from sklearn.metrics import roc_auc_score
        print(roc_auc_score(y, pred.detach().numpy()))

    if party == "guest":

        ctx = create_ctx(guest, get_current_datetime_str())
        df = pd.read_csv('/home/cwj/FATE/FATE-2.0/FATE/examples/data/breast_hetero_guest.csv')
        X_g = t.Tensor(df.drop(columns=['id', 'y']).values).type(t.float64)[0: sample_num]
        y = t.Tensor(df['y'].values).type(t.float64)[0: sample_num]

        dataset = TensorDataset(X_g, y)

        agg_layer = AggLayerGuest(merge_type='concat')
        loss_fn = t.nn.BCELoss()

        model = HeteroNNModelGuest(
            top_model=guest_top,
            bottom_model=guest_bottom,
            agg_layer=agg_layer
        )
        model.set_context(ctx)

        optimizer = t.optim.Adam(model.parameters(), lr=0.01)

        for i in range(epoch):
            loss_sum = 0
            batch_idx = 0
            for x_, y_ in tqdm.tqdm(DataLoader(dataset, batch_size=batch_size)):
                optimizer.zero_grad()
                fw = model(x_)
                loss_ = loss_fn(fw.flatten(), y_)
                model.backward(loss_)
                optimizer.step()
                loss_sum += loss_.item()
                batch_idx += 1
            print(loss_sum / batch_idx)

        pred = model(X_g)
        from sklearn.metrics import roc_auc_score
        print(roc_auc_score(y, pred.detach()))

    elif party == "host":

        ctx = create_ctx(host, get_current_datetime_str())

        df = pd.read_csv('/home/cwj/FATE/FATE-2.0/FATE/examples/data/breast_hetero_host.csv')
        X_h = t.Tensor(df.drop(columns=['id']).values).type(t.float64)[0: sample_num]

        dataset = TensorDataset(X_h)

        layer = AggLayerHost()
        model = HeteroNNModelHost(agg_layer=layer, bottom_model=host_bottom)
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)
        model.set_context(ctx)

        for i in range(epoch):
            for x_ in tqdm.tqdm(DataLoader(dataset, batch_size=batch_size)):
                optimizer.zero_grad()
                model.forward(x_[0])
                model.backward()
                optimizer.step()

        pred = model(X_h)
