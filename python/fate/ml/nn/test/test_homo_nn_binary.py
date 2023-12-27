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

from fate.ml.nn.homo.fedavg import FedAVGClient, FedArguments, TrainingArguments, FedAVGServer
import torch as t
from fate.ml.nn.dataset.table import TableDataset
import sys


arbiter = ("arbiter", 10000)
guest = ("guest", 10000)
host = ("host", 9999)
name = "fed"


def create_ctx(local):
    from fate.arch import Context
    from fate.arch.computing.backends.standalone import CSession
    from fate.arch.federation.backends.standalone import StandaloneFederation
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    computing = CSession(data_dir="./session_dir")
    return Context(
        computing=computing, federation=StandaloneFederation(computing, name, local, [guest, host, arbiter])
    )


if __name__ == "__main__":
    epoch = 10
    model = t.nn.Sequential(t.nn.Linear(30, 1), t.nn.Sigmoid())

    ds = TableDataset(return_dict=False, to_tensor=True)
    ds.load("./../../../../../examples/data/breast_homo_guest.csv")

    ds_val = TableDataset(return_dict=False, to_tensor=True)
    ds_val.load("./../../../../../examples/data/breast_homo_test.csv")

    if sys.argv[1] == "guest":
        ctx = create_ctx(guest)
        fed_args = FedArguments(aggregate_strategy="epoch", aggregate_freq=1, aggregator="secure_aggregate")
        args = TrainingArguments(num_train_epochs=5, per_device_train_batch_size=16)
        trainer = FedAVGClient(
            ctx=ctx,
            model=model,
            fed_args=fed_args,
            training_args=args,
            loss_fn=t.nn.BCELoss(),
            optimizer=t.optim.SGD(model.parameters(), lr=0.01),
            train_set=ds,
            val_set=ds_val,
        )
        trainer.train()

    elif sys.argv[1] == "host":
        ctx = create_ctx(host)
        fed_args = FedArguments(aggregate_strategy="epoch", aggregate_freq=1, aggregator="secure_aggregate")
        args = TrainingArguments(num_train_epochs=5, per_device_train_batch_size=16)
        trainer = FedAVGClient(
            ctx=ctx,
            model=model,
            fed_args=fed_args,
            training_args=args,
            loss_fn=t.nn.BCELoss(),
            optimizer=t.optim.SGD(model.parameters(), lr=0.01),
            train_set=ds,
            val_set=ds_val,
        )
        trainer.train()

    else:
        ctx = create_ctx(arbiter)
        trainer = FedAVGServer(ctx)
        trainer.train()
