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
    guest_bottom = t.nn.Linear(10, 4)
    guest_top = t.nn.Sequential(
                                 t.nn.Linear(4, 1),
                                 t.nn.Sigmoid()
                               )
    host_bottom = t.nn.Linear(20, 4)

    # # make random fake data
    sample_num = 569

    if party == "guest":

        from fate.ml.evaluation.metric_base import MetricEnsemble
        from fate.ml.evaluation.classification import MultiAccuracy
        ctx = create_ctx(guest, get_current_datetime_str())
        ctx.mpc.init()
        df = pd.read_csv('./../../../../../examples/data/breast_hetero_guest.csv')
        X_g = t.Tensor(df.drop(columns=['id', 'y']).values)[0: sample_num]
        y = t.Tensor(df['y'].values)[0: sample_num].reshape((-1, 1))

        dataset = TensorDataset(X_g, y)
        loss_fn = t.nn.BCELoss()

        model = HeteroNNModelGuest(
            top_model=guest_top,
            bottom_model=guest_bottom,
            agglayer_arg=SSHEArgument(
                guest_in_features=4,
                host_in_features=4,
                out_features=4,
                layer_lr=0.01
            )
        )
        model
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)

        args = TrainingArguments(
            num_train_epochs=5,
            per_device_train_batch_size=16,
            no_cuda=True,
            disable_tqdm=False
        )
        trainer = HeteroNNTrainerGuest(
            ctx=ctx,
            model=model,
            optimizer=optimizer,
            train_set=dataset,
            val_set=dataset,
            loss_fn=loss_fn,
            training_args=args,
            compute_metrics=MetricEnsemble().add_metric(MultiAccuracy())
        )
        trainer.train()
        pred = trainer.predict(dataset)
        # # compute auc
        from sklearn.metrics import roc_auc_score
        print(roc_auc_score(pred.label_ids, pred.predictions))

    elif party == "host":

        ctx = create_ctx(host, get_current_datetime_str())
        ctx.mpc.init()
        df = pd.read_csv('./../../../../../examples/data/breast_hetero_host.csv')
        X_h = t.Tensor(df.drop(columns=['id']).values)[0: sample_num]

        dataset = TensorDataset(X_h)

        model = HeteroNNModelHost(
            bottom_model=host_bottom,
            agglayer_arg=SSHEArgument(
                guest_in_features=4,
                host_in_features=4,
                out_features=4,
                layer_lr=0.01
            )
        )
        optimizer = t.optim.Adam(model.parameters(), lr=0.01)

        args = TrainingArguments(
            num_train_epochs=5,
            per_device_train_batch_size=16,
            no_cuda=True,
            disable_tqdm=False
        )

        trainer = HeteroNNTrainerHost(
            ctx=ctx,
            model=model,
            optimizer=optimizer,
            train_set=dataset,
            val_set=dataset,
            training_args=args
        )
        trainer.train()
        trainer.predict(dataset)

