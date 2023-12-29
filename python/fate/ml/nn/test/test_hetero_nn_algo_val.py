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

from fate.ml.nn.model_zoo.hetero_nn_model import HeteroNNModelGuest, HeteroNNModelHost
from fate.ml.nn.hetero.hetero_nn import HeteroNNTrainerGuest, HeteroNNTrainerHost, TrainingArguments
from fate.ml.nn.model_zoo.agg_layer.agg_layer import AggLayerGuest, AggLayerHost
import sys
from datetime import datetime
import pandas as pd
from torch.utils.data import TensorDataset


def get_current_datetime_str():
    return datetime.now().strftime("%Y-%m-%d-%H-%M")


guest = ("guest", "10000")
host = ("host", "9999")
name = get_current_datetime_str()


def create_ctx(local, context_name):
    from fate.arch import Context
    from fate.arch.computing.backends.standalone import CSession
    from fate.arch.federation.backends.standalone import StandaloneFederation
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
    return Context(computing=computing, federation=StandaloneFederation(computing, context_name, local, [guest, host]))


if __name__ == "__main__":
    party = sys.argv[1]
    import torch as t

    def set_seed(seed):
        t.manual_seed(seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(seed)
            t.backends.cudnn.deterministic = True
            t.backends.cudnn.benchmark = False

    set_seed(42)

    batch_size = 64
    epoch = 10
    guest_bottom = t.nn.Linear(10, 4).double()
    guest_top = t.nn.Sequential(t.nn.Linear(4, 1), t.nn.Sigmoid()).double()
    host_bottom = t.nn.Linear(20, 4).double()

    # # make random fake data
    sample_num = 569

    if party == "guest":
        from fate.ml.evaluation.classification import AUC, BinaryAccuracy
        from fate.ml.evaluation.metric_base import MetricEnsemble

        ctx = create_ctx(guest, get_current_datetime_str())
        df = pd.read_csv("./../../../../../examples/data/breast_hetero_guest.csv")
        X_g = t.Tensor(df.drop(columns=["id", "y"]).values).type(t.float64)[0:sample_num]
        y = t.Tensor(df["y"].values).type(t.float64)[0:sample_num].reshape((-1, 1))

        dataset = TensorDataset(X_g, y)

        agglayer = AggLayerGuest()
        loss_fn = t.nn.BCELoss()

        model = HeteroNNModelGuest(top_model=guest_top, agg_layer=agglayer, bottom_model=guest_bottom)
        model.set_context(ctx)
        model.double()
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)

        args = TrainingArguments(
            num_train_epochs=5,
            per_device_train_batch_size=16,
            disable_tqdm=True,
            no_cuda=True,
            eval_steps=1,
            evaluation_strategy="epoch",
        )

        metrics = MetricEnsemble().add_metric(AUC()).add_metric(BinaryAccuracy())

        trainer = HeteroNNTrainerGuest(
            ctx=ctx,
            model=model,
            optimizer=optimizer,
            train_set=dataset,
            val_set=dataset,
            loss_fn=loss_fn,
            training_args=args,
            compute_metrics=metrics,
        )
        trainer.train()
        pred = trainer.predict(dataset)
        print(metrics(pred))

    elif party == "host":
        ctx = create_ctx(host, get_current_datetime_str())
        df = pd.read_csv("./../../../../../examples/data/breast_hetero_host.csv")
        X_h = t.Tensor(df.drop(columns=["id"]).values).type(t.float64)[0:sample_num]

        dataset = TensorDataset(X_h)

        layer = AggLayerHost()
        model = HeteroNNModelHost(bottom_model=host_bottom, agg_layer=layer)
        model.double()
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)

        args = TrainingArguments(
            num_train_epochs=5,
            per_device_train_batch_size=16,
            disable_tqdm=True,
            no_cuda=True,
            eval_steps=1,
            evaluation_strategy="epoch",
        )

        trainer = HeteroNNTrainerHost(
            ctx=ctx,
            model=model,
            optimizer=optimizer,
            train_set=dataset,
            val_set=dataset,
            training_args=args,
        )
        trainer.train()
        trainer.predict(dataset)
