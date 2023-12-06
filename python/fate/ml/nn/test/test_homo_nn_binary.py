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
    from fate.arch.computing.standalone import CSession
    from fate.arch.federation.standalone import StandaloneFederation
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    computing = CSession()
    return Context(
        computing=computing, federation=StandaloneFederation(computing, name, local, [guest, host, arbiter])
    )


if __name__ == "__main__":
    epoch = 10
    model = t.nn.Sequential(t.nn.Linear(30, 1), t.nn.Sigmoid())

    ds = TableDataset(return_dict=False, to_tensor=True)
    ds.load('./../../../../../examples/data/breast_homo_guest.py')

    if sys.argv[1] == "guest":
        ctx = create_ctx(guest)
        fed_args = FedArguments(aggregate_strategy="epoch", aggregate_freq=1, aggregator="secure_aggregate")
        args = TrainingArguments(
            num_train_epochs=5, per_device_train_batch_size=16
        )
        trainer = FedAVGClient(
            ctx=ctx,
            model=model,
            fed_args=fed_args,
            training_args=args,
            loss_fn=t.nn.BCELoss(),
            optimizer=t.optim.SGD(model.parameters(), lr=0.01),
            train_set=ds
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
        )
        trainer.train()

    else:
        ctx = create_ctx(arbiter)
        trainer = FedAVGServer(ctx)
        trainer.train()
