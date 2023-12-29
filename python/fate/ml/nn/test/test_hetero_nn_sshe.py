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
from fate.ml.nn.model_zoo.hetero_nn_model import SSHEArgument
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
    computing = CSession(data_dir="./cession_dir")
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

    batch_size = 64
    epoch = 10
    guest_bottom = t.nn.Linear(10, 10)
    guest_top = t.nn.Sequential(t.nn.Linear(10, 1), t.nn.Sigmoid())
    host_bottom = t.nn.Linear(20, 10)

    # # make random fake data
    sample_num = 569

    args = TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=256,
        logging_strategy="epoch",
        no_cuda=True,
        log_level="debug",
        disable_tqdm=False,
    )

    if party == "guest":
        from fate.ml.evaluation.metric_base import MetricEnsemble
        from fate.ml.evaluation.classification import MultiAccuracy

        ctx = create_ctx(guest, get_current_datetime_str())
        ctx.mpc.init()
        df = pd.read_csv("./../../../../../examples/data/breast_hetero_guest.csv")
        X_g = t.Tensor(df.drop(columns=["id", "y"]).values)[0:sample_num]
        y = t.Tensor(df["y"].values)[0:sample_num].reshape((-1, 1))

        dataset = TensorDataset(X_g, y)
        loss_fn = t.nn.BCELoss()

        model = HeteroNNModelGuest(
            top_model=guest_top,
            bottom_model=guest_bottom,
            agglayer_arg=SSHEArgument(guest_in_features=10, host_in_features=10, out_features=10),
        )
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)

        trainer = HeteroNNTrainerGuest(
            ctx=ctx,
            model=model,
            optimizer=optimizer,
            train_set=dataset,
            val_set=dataset,
            loss_fn=loss_fn,
            training_args=args,
            compute_metrics=MetricEnsemble().add_metric(MultiAccuracy()),
        )
        trainer.train()
        pred_0 = trainer.predict(dataset)
        # # compute auc
        from sklearn.metrics import roc_auc_score

        print(roc_auc_score(pred_0.label_ids, pred_0.predictions))

        pred_1 = trainer.predict(dataset)
        # # compute auc
        from sklearn.metrics import roc_auc_score

        print(roc_auc_score(pred_1.label_ids, pred_1.predictions))

    elif party == "host":
        ctx = create_ctx(host, get_current_datetime_str())
        ctx.mpc.init()
        df = pd.read_csv("./../../../../../examples/data/breast_hetero_host.csv")
        X_h = t.Tensor(df.drop(columns=["id"]).values)[0:sample_num]

        dataset = TensorDataset(X_h)

        model = HeteroNNModelHost(
            bottom_model=host_bottom,
            agglayer_arg=SSHEArgument(guest_in_features=10, host_in_features=10, out_features=10),
        )
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)

        trainer = HeteroNNTrainerHost(
            ctx=ctx, model=model, optimizer=optimizer, train_set=dataset, val_set=dataset, training_args=args
        )
        trainer.train()
        trainer.predict(dataset)
        trainer.predict(dataset)
