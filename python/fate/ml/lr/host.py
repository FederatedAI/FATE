import logging

import numpy as np
import torch
from fate.arch import tensor
from fate.arch.dataframe import CSVReader, DataLoader
from fate.interface import Context, ModelsLoader, ModelsSaver
from pandas import pandas

from ..abc.module import HeteroModule

logger = logging.getLogger(__name__)


class DataframeMock:
    def __init__(self, ctx) -> None:
        guest_data_path = "/Users/sage/proj/FATE/2.0.0-alpha/" \
                          "examples/data/breast_hetero_host.csv"
        self.data = CSVReader(
            id_name="id",
            delimiter=",",
            dtype="float32"
        ).to_frame(ctx, guest_data_path)
        self.num_features = 20
        self.num_sample = len(self.data)

    def batches(self, batch_size):
        num_batchs = (self.num_sample - 1) // batch_size + 1
        for chunk in np.array_split(self.data, num_batchs):
            yield tensor.tensor(torch.Tensor(chunk[:, 1:]))


class LrModuleHost(HeteroModule):
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
        self.learning_rate = 0.1
        self.alpha = 1.0
        self.batch_size = 1000

    def fit(self, ctx: Context, train_data) -> None:
        # mock data
        train_data = DataframeMock(ctx)
        batch_loader = DataLoader(train_data.data, ctx=ctx, batch_size=self.batch_size,
                                  mode="hetero", role="host")
        # get encryptor
        encryptor = ctx.arbiter("encryptor").get()

        w = tensor.tensor(
            torch.randn((train_data.num_features, 1), dtype=torch.float32)
        )
        for i, iter_ctx in ctx.range(self.max_iter):
            logger.info(f"start iter {i}")
            j = 0
            for batch_ctx, X in iter_ctx.iter(batch_loader):
                h = X.shape[0]
                logger.info(f"start batch {j}")
                Xw_h = 0.25 * tensor.matmul(X, w)
                encryptor.encrypt(Xw_h).to(batch_ctx.guest, "Xw_h")
                encryptor.encrypt(tensor.matmul(Xw_h.T, Xw_h)).to(
                    batch_ctx.guest, "Xw2_h"
                )
                d = batch_ctx.guest.get("d")
                tensor.matmul(X.T, d).to(batch_ctx.arbiter, "g_enc")
                g = batch_ctx.arbiter.get("g")
                g += self.alpha * w
                w -= (self.learning_rate / h) * g
                logger.info(f"w={w}")
                j += 1
