import logging
from typing import Union

from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.arch.protocol.mpc.nn.sshe.lr_layer import (
    SSHELogisticRegressionLayer,
    SSHELogisticRegressionLossLayer,
    SSHEOptimizerSGD,
)
from ..abc.module import Module, Model, HeteroModule

logger = logging.getLogger(__name__)


class SSHELogisticRegression(Module):
    def __init__(self, epochs, batch_size, tol, early_stop, learning_rate, init_param,
                 encrypted_reveal=True, threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.tol = tol
        self.early_stop = early_stop
        self.learning_rate = learning_rate
        self.init_param = init_param
        self.threshold = threshold
        self.encrypted_reveal = encrypted_reveal

        self.estimator = None
        self.ovr = False
        self.labels = None

    def fit(self, ctx: Context, train_data: DataFrame, validate_data=None):
        train_data_binarized_label = train_data.label.get_dummies()
        label_count = train_data_binarized_label.shape[1]
        ctx.hosts.put("label_count", label_count)

    def get_model(self):
        all_estimator = {}
        if self.ovr:
            for label, estimator in self.estimator.items():
                all_estimator[label] = estimator.get_model()
        else:
            all_estimator = self.estimator.get_model()
        return {
            "data": {"estimator": all_estimator},
            "meta": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "init_param": self.init_param,
                "optimizer_param": self.optimizer_param,
                "labels": self.labels,
                "ovr": self.ovr,
                "threshold": self.threshold,
                "encrypted_reveal": self.encrypted_reveal,
            },
        }

    def from_model(cls, model: Union[dict, Model]):
        pass


class SSHELREstimator(HeteroModule):
    def __init__(self, epochs=None, batch_size=None, optimizer=None, learning_rate=None, init_param=None,
                 encrypted_reveal=True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = learning_rate
        self.init_param = init_param
        self.encrypted_reveal = encrypted_reveal

        self.w = None
        self.start_epoch = 0
        self.end_epoch = -1
        self.is_converged = False
        self.header = None

    def fit_binary_model(self, ctx: Context, train_data: DataFrame) -> None:

        rank_a, rank_b = ctx.hosts[0].rank, ctx.guest.rank
        y = ctx.mpc.cond_call(lambda: train_data.label.as_tensor(), lambda: None, dst=rank_b)
        h = train_data.as_tensor()
        # generator = torch.Generator().manual_seed(0)
        layer = SSHELogisticRegressionLayer(
            ctx,
            in_features_a=ctx.mpc.option_call(lambda: h.shape[1], dst=rank_a),
            in_features_b=ctx.mpc.option_call(lambda: h.shape[1], dst=rank_b),
            out_features=1,
            rank_a=rank_a,
            rank_b=rank_b,
            # generator=generator,
        )
        loss_fn = SSHELogisticRegressionLossLayer(ctx, rank_a=rank_a, rank_b=rank_b)
        optimizer = SSHEOptimizerSGD(ctx, layer.parameters(), lr=self.lr)

        for i in range(20):
            # mpc encrypted [wx]
            # to get decrypted wx: z.get_plain_text()
            z = layer(h)
            loss = loss_fn(z, y)
            if i % 3 == 0:
                logger.info(f"loss: {loss.get()}")
            loss.backward()
            optimizer.step()
