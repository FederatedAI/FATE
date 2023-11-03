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

        def __init__(self, guest_b, guest_t, host_b, guest_i, host_i):
            super(HeteroNNLocalModel, self).__init__()
            self._guest_b = guest_b
            self._guest_t = guest_t
            self._host_b = host_b
            self._guest_i = guest_i
            self._host_i = host_i
        def forward(self, x_g, x_h):
            fw_g = self._guest_i(self._guest_b(x_g))
            fw_h = self._host_i(self._host_b(x_h))
            fw_ = fw_g + fw_h
            fw_ = t.nn.ReLU()(fw_)
            fw_ = self._guest_t(fw_)
            return fw_

    set_seed(42)

    batch_size = 64
    epoch = 10
    guest_bottom = t.nn.Linear(10, 4).double()
    guest_top = t.nn.Sequential(
                                 t.nn.Linear(8, 1),
                                 t.nn.Sigmoid()
                               ).double()
    host_bottom = t.nn.Linear(20, 8).double()

    # # make random fake data
    sample_num = 569

    if party == "guest":

        ctx = create_ctx(guest, get_current_datetime_str())
        df = pd.read_csv('/home/cwj/FATE/FATE-2.0/FATE/examples/data/breast_hetero_guest.csv')
        y = t.Tensor(df['y'].values).type(t.float64)[0: sample_num]

        dataset = TensorDataset(y)
        loss_fn = t.nn.BCELoss()

        model = HeteroNNModelGuest(
            top_model=guest_top,
            agg_layer=AggLayerGuest()
        )
        model.set_context(ctx)

        optimizer = t.optim.Adam(model.parameters(), lr=0.01)

        for i in range(epoch):
            loss_sum = 0
            batch_idx = 0
            for y_ in tqdm.tqdm(DataLoader(dataset, batch_size=batch_size)):
                y_ = y_[0]
                optimizer.zero_grad()
                fw = model(None)  # no guest X
                loss_ = loss_fn(fw.flatten(), y_)
                model.backward(loss_)
                optimizer.step()
                loss_sum += loss_.item()
                batch_idx += 1
            print(loss_sum / batch_idx)

        pred = model.predict()
        # compute auc
        from sklearn.metrics import roc_auc_score
        print(roc_auc_score(y, pred))

    elif party == "host":

        ctx = create_ctx(host, get_current_datetime_str())

        df = pd.read_csv('/home/cwj/FATE/FATE-2.0/FATE/examples/data/breast_hetero_host.csv')
        X_h = t.Tensor(df.drop(columns=['id']).values).type(t.float64)[0: sample_num]

        dataset = TensorDataset(X_h)

        layer = AggLayerHost()
        model = HeteroNNModelHost(agg_layer=AggLayerHost(), bottom_model=host_bottom)
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)
        model.set_context(ctx)

        for i in range(epoch):
            for x_ in DataLoader(dataset, batch_size=batch_size):
                optimizer.zero_grad()
                model.forward(x_[0])
                model.backward()
                optimizer.step()

        pred = model.predict(X_h)