from fate.ml.nn.hetero.model.guest import SplitNNGuest
from fate.ml.nn.hetero.model.host import SplitNNHost
from fate.ml.nn.hetero.agg_layer.plaintext_agg_layer import InteractiveLayerGuest, InteractiveLayerHost
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

    batch_size = 569
    epoch = 2
    guest_bottom = t.nn.Linear(10, 4).double()
    guest_top = t.nn.Sequential(
                                 t.nn.Linear(4, 1),
                                 t.nn.Sigmoid()
                               ).double()
    host_bottom = t.nn.Linear(20, 4).double()

    # # make random fake data
    sample_num = 569

    if party == 'local':

        ctx = create_ctx(guest, get_current_datetime_str())

        df = pd.read_csv('/examples/data/breast_hetero_guest.csv')
        X_g = t.Tensor(df.drop(columns=['id', 'y']).values).type(t.float64)[0: sample_num]
        y = t.Tensor(df['y'].values).type(t.float64)[0: sample_num]
        df = pd.read_csv('/examples/data/breast_hetero_host.csv')
        X_h = t.Tensor(df.drop(columns=['id']).values).type(t.float64)[0: sample_num]

        interactive_layer = InteractiveLayerGuest(ctx, 4, 4, 4)
        local_model = HeteroNNLocalModel(guest_bottom, guest_top, host_bottom,
                                         interactive_layer._guest_model.double(),
                                         interactive_layer._host_model[0].double())
        loss_fn = t.nn.BCELoss()
        optimizer = t.optim.SGD(local_model.parameters(), lr=0.01)
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

    if party == "guest":

        ctx = create_ctx(guest, get_current_datetime_str())
        df = pd.read_csv('/examples/data/breast_hetero_guest.csv')
        X_g = t.Tensor(df.drop(columns=['id', 'y']).values).type(t.float64)[0: sample_num]
        y = t.Tensor(df['y'].values).type(t.float64)[0: sample_num]

        dataset = TensorDataset(X_g, y)

        interactive_layer = InteractiveLayerGuest(ctx, 4,4,4)
        interactive_layer._guest_model = interactive_layer._guest_model.double()
        interactive_layer._host_model[0] = interactive_layer._host_model[0].double()
        loss_fn = t.nn.BCELoss()

        model = SplitNNGuest(
            top_model=guest_top,
            bottom_model=guest_bottom,
            interactive_layer=interactive_layer,
            ctx=ctx
        )

        optimizer = t.optim.SGD(model.parameters(), lr=0.01)

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


    elif party == "host":

        ctx = create_ctx(host, get_current_datetime_str())

        df = pd.read_csv('/examples/data/breast_hetero_host.csv')
        X_h = t.Tensor(df.drop(columns=['id']).values).type(t.float64)[0: sample_num]

        dataset = TensorDataset(X_h)

        layer = InteractiveLayerHost(ctx)
        model = SplitNNHost(host_bottom, interactive_layer=layer, ctx=ctx)
        optimizer = t.optim.SGD(model.parameters(), lr=0.01)

        for i in range(epoch):
            for x_ in DataLoader(dataset, batch_size=batch_size):
                optimizer.zero_grad()
                model.forward(x_[0])
                model.backward()
                optimizer.step()

        pred = model(X_h)