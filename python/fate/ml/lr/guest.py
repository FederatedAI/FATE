import logging

import numpy as np
import torch
from fate.arch import tensor
from fate.interface import Context, ModelsLoader, ModelsSaver
from fate.arch.dataframe import CSVReader, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.utils.fixes import sklearn

from ..abc.module import HeteroModule

logger = logging.getLogger(__name__)


class GuestDataframeMock:
    def __init__(self, ctx) -> None:
        guest_data_path = "/Users/sage/proj/FATE/2.0.0-alpha/" \
                          "examples/data/breast_hetero_guest.csv"
        self.data = CSVReader(
            id_name="id",
            label_name="y",
            label_type="float32",
            delimiter=",",
            dtype="float32"
        ).to_frame(ctx, guest_data_path)
        self.num_features = 10
        self.num_sample = len(self.data)

    def batches(self, batch_size):
        num_batchs = (self.num_sample - 1) // batch_size + 1
        for chunk in np.array_split(self.data, num_batchs):
            yield (
                tensor.tensor(torch.Tensor(chunk[:, 2:])),
                2 * tensor.tensor(torch.Tensor(chunk[:, 1:2]) - 1),
            )


class LrModuleGuest(HeteroModule):
    def __init__(
        self,
        penalty="l2",
        *,
        dual=False,
        tol=1e-4,
        C=1.0,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=None,
        solver="lbfgs",
        max_iter=100,
        multi_class="auto",
        verbose=0,
        warm_start=False,
        n_jobs=None,
        l1_ratio=None,
    ):
        self.max_iter = max_iter
        self.batch_size = 100
        self.learning_rate = 0.1
        self.alpha = 1.0

    def fit(self, ctx: Context, train_data) -> None:
        """
        l(w) = 1/h * Σ(log(2) - 0.5 * y * xw + 0.125 * (wx)^2)
        ∇l(w) = 1/h * Σ(0.25 * xw - 0.5 * y)x = 1/h * Σdx
        where d = 0.25(xw - 2y)
        loss = log2 - (1/N)*0.5*∑ywx + (1/N)*0.125*[∑(Wg*Xg)^2 + ∑(Wh*Xh)^2 + 2 * ∑(Wg*Xg * Wh*Xh)]
        """
        # mock data
        train_data = GuestDataframeMock(ctx)
        batch_loader = DataLoader(train_data.data, ctx=ctx, batch_size=self.batch_size,
                                  mode="hetero", role="guest", sync_arbiter=True)
        # get encryptor
        ctx.arbiter("encryptor").get()
        logger.info(train_data.num_sample)

        w = tensor.tensor(
            torch.randn((train_data.num_features, 1), dtype=torch.float32)
        )
        for i, iter_ctx in ctx.range(self.max_iter):
            logger.info(f"start iter {i}")
            j = 0
            for batch_ctx, (X, Y) in iter_ctx.iter(batch_loader):
                h = X.shape[0]

                # d
                Xw = tensor.matmul(X, w)
                d = 0.25 * Xw - 0.5 * Y
                loss = 0.125 / h * tensor.matmul(Xw.T, Xw) - 0.5 / h * tensor.matmul(
                    Xw.T, Y
                )
                for Xw_h in batch_ctx.hosts.get("Xw_h"):
                    d += Xw_h
                    loss -= 0.5 / h * tensor.matmul(Y.T, Xw_h)
                    loss += 0.25 / h * tensor.matmul(Xw.T, Xw_h)
                for Xw2_h in batch_ctx.hosts.get("Xw2_h"):
                    loss += 0.125 / h * Xw2_h
                batch_ctx.hosts.put(d=d)
                batch_ctx.arbiter.put(loss=loss)

                # gradian
                batch_ctx.arbiter.put("g_enc", X.T @ d)
                g: tensor.Tensor = batch_ctx.arbiter.get("g")
                # apply l2 penalty
                g += self.alpha * w
                w -= (self.learning_rate / h) * g
                logger.info(f"w={w}")
                j += 1
