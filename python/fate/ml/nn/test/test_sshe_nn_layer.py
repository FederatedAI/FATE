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
    computing = CSession(data_dir="./cession_dir")
    return Context(computing=computing, federation=StandaloneFederation(computing, context_name, local, [guest, host]))


if __name__ == "__main__":
    party = sys.argv[1]

    def set_seed(seed):
        t.manual_seed(seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(seed)
            t.backends.cudnn.deterministic = True
            t.backends.cudnn.benchmark = False

    from fate.arch.protocol.mpc.nn.sshe.nn_layer import SSHENeuralNetworkAggregatorLayer, SSHENeuralNetworkOptimizerSGD

    set_seed(42)

    sample_num = 569
    in_features_a = 4
    in_features_b = 4
    out_features = 4
    lr = 0.05

    if party == "guest":
        ctx = create_ctx(guest, get_current_datetime_str())
        ctx.mpc.init()

        df = pd.read_csv("./../../../../../examples/data/breast_hetero_guest.csv")
        X_g = t.Tensor(df.drop(columns=["id", "y"]).values)[0:sample_num]
        y = t.Tensor(df["y"].values)[0:sample_num].reshape((-1, 1))

        bottom_model = t.nn.Sequential(t.nn.Linear(10, 4), t.nn.ReLU())

        top_model = t.nn.Sequential(t.nn.Linear(4, 1), t.nn.Sigmoid())

        from fate.ml.nn.model_zoo.agg_layer.sshe.agg_layer import SSHEAggLayerGuest

        layer = SSHEAggLayerGuest(
            guest_in_features=in_features_a, host_in_features=in_features_b, out_features=out_features, layer_lr=lr
        )
        layer.set_context(ctx)
        t_optimizer = torch.optim.SGD(list(bottom_model.parameters()) + list(top_model.parameters()), lr=lr)

        fw_rs = top_model(layer(bottom_model(X_g)))

        loss = t.nn.BCELoss()(fw_rs, y)
        t_optimizer.zero_grad()
        loss.backward()
        layer.step()
        t_optimizer.step()

    if party == "host":
        ctx = create_ctx(host, get_current_datetime_str())
        ctx.mpc.init()

        df = pd.read_csv("./../../../../../examples/data/breast_hetero_host.csv")
        X_h = t.Tensor(df.drop(columns=["id"]).values)[0:sample_num]

        bottom_model = t.nn.Sequential(t.nn.Linear(20, 4), t.nn.ReLU())

        from fate.ml.nn.model_zoo.agg_layer.sshe.agg_layer import SSHEAggLayerHost

        layer = SSHEAggLayerHost(
            guest_in_features=in_features_a, host_in_features=in_features_b, out_features=out_features, layer_lr=lr
        )
        t_optimizer = t.optim.Adam(bottom_model.parameters(), lr=lr)
        layer.set_context(ctx)
        fw = layer(bottom_model(X_h))
        fw.backward()
        layer.step()
        t_optimizer.step()
