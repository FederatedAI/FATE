import sys
import torch
import torch as t
from datetime import datetime
import pandas as pd


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
    computing = CSession(data_dir='./cession_dir')
    return Context(
        computing=computing, federation=StandaloneFederation(computing, context_name, local, [guest, host])
    )


if __name__ == '__main__':

    party = sys.argv[1]
    mode = sys.argv[2]
    epoch = 1

    if party == 'guest':
        ctx = create_ctx(local=guest, context_name=name)
    else:
        ctx = create_ctx(local=host, context_name=name)

    from fate.arch.protocol.mpc.nn.sshe.nn_layer import SSHENeuralNetworkAggregatorLayer, SSHENeuralNetworkOptimizerSGD
    def set_seed(seed):
        t.manual_seed(seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(seed)
            t.backends.cudnn.deterministic = True
            t.backends.cudnn.benchmark = False

    set_seed(114514)

    sample_num = 5
    df = pd.read_csv('/home/cwj/FATE/FATE-2.0/FATE/examples/data/breast_hetero_guest.csv')
    X_g = t.Tensor(df.drop(columns=['id', 'y']).values)[0: sample_num]
    y = t.Tensor(df['y'].values).reshape((-1, 1))[0: sample_num]
    df = pd.read_csv('/home/cwj/FATE/FATE-2.0/FATE/examples/data/breast_hetero_host.csv')
    X_h = t.Tensor(df.drop(columns=['id']).values)[0: sample_num]

    lr = 0.01

    # bottom model
    guest_bottom = t.nn.Linear(10, 4)
    host_bottom = t.nn.Linear(20, 8)

    # sshe
    guest_wa = t.nn.Linear(8, 4, bias=False).weight / 2
    host_wa = t.nn.Linear(8, 4, bias=False).weight / 2

    guest_wb = t.nn.Linear(4, 4, bias=False).weight / 2
    host_wb = t.nn.Linear(4, 4, bias=False).weight / 2

    complete_guest = guest_wb + host_wb
    complete_host = guest_wa + host_wa

    sshe_guest = t.nn.Linear(4, 4, bias=False)
    sshe_host = t.nn.Linear(8, 4, bias=False)
    sshe_guest.weight.data = complete_guest
    sshe_host.weight.data = complete_host

    # top model
    top_model = t.nn.Sequential(
        t.nn.Linear(4, 1),
        t.nn.Sigmoid()
    )

    if mode == 'local':
        # add all model parameter into SGD optimizer
        optimizer = t.optim.SGD(list(guest_bottom.parameters()) + list(host_bottom.parameters()) + list(top_model.parameters()) \
                                + list(sshe_guest.parameters()) + list(sshe_host.parameters()), lr=lr)

        def forward():
            guest_bottom_out = guest_bottom(X_g)
            host_bottom_out = host_bottom(X_h)
            guest_sshe_out = sshe_guest(guest_bottom_out)
            host_sshe_out = sshe_host(host_bottom_out)
            guest_top_out = top_model(guest_sshe_out + host_sshe_out)
            return guest_top_out, guest_sshe_out, host_sshe_out, guest_bottom_out, host_bottom_out

        pred_ = None
        for i in range(epoch):
            out_, a, b, c, d = forward()
            pred_ = out_.detach().numpy()
            loss = t.nn.BCELoss()(out_, y)
            loss.backward()
            optimizer.step()
            print(f"loss={loss}")

        # compute auc here:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y, pred_)
        print(auc)

    else:
        ctx.mpc.init()
        if party == 'guest':
            layer = SSHENeuralNetworkAggregatorLayer(
                ctx,
                in_features_a=8,
                in_features_b=4,
                out_features=4,
                rank_a=ctx.hosts[0].rank,
                rank_b=ctx.guest.rank,
                wa_init_fn=lambda shape: complete_host.T,
                wb_init_fn=lambda shape: complete_guest.T,
            )

            optimizer_sshe = SSHENeuralNetworkOptimizerSGD(ctx, layer.parameters(), lr=lr)
            optimizer_other = t.optim.SGD(list(guest_bottom.parameters()) + list(top_model.parameters()), lr=lr)

            def forward():
                guest_bottom_out = guest_bottom(X_g)
                sshe_out = layer(guest_bottom_out)
                top_out = top_model(sshe_out)
                return top_out, sshe_out, guest_bottom_out

            for i in range(epoch):
                optimizer_other.zero_grad()
                out_, sshe_out, guest_b_out = forward()
                loss = t.nn.BCELoss()(out_, y)
                loss.backward()
                print(loss)
                optimizer_sshe.step()
                optimizer_other.step()

        else:
            layer = SSHENeuralNetworkAggregatorLayer(
                ctx,
                in_features_a=8,
                in_features_b=4,
                out_features=4,
                rank_a=ctx.hosts[0].rank,
                rank_b=ctx.guest.rank,
                wa_init_fn=lambda shape: complete_host.T,
                wb_init_fn=lambda shape: complete_guest.T,
            )

            optimizer_ss = SSHENeuralNetworkOptimizerSGD(ctx, layer.parameters(), lr=lr)
            optimizer_other = t.optim.SGD(list(host_bottom.parameters()), lr=lr)

            def forward():
                host_bottom_out = host_bottom(X_h)
                sshe_out = layer(host_bottom_out)
                return sshe_out, host_bottom_out

            for i in range(epoch):
                optimizer_other.zero_grad()
                out_, host_b_out = forward()
                out_.backward()
                optimizer_ss.step()
                optimizer_other.step()


