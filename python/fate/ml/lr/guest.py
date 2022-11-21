import logging

import torch
from fate.arch import dataframe, tensor
from fate.interface import Context

from ..abc.module import HeteroModule

logger = logging.getLogger(__name__)


class LrModuleGuest(HeteroModule):
    def __init__(
        self,
        max_iter,
        batch_size,
        learning_rate=0.01,
        alpha=1.0,
    ):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.alpha = alpha

        self.w = None

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        """
        l(w) = 1/h * Σ(log(2) - 0.5 * y * xw + 0.125 * (wx)^2)
        ∇l(w) = 1/h * Σ(0.25 * xw - 0.5 * y)x = 1/h * Σdx
        where d = 0.25(xw - 2y)
        loss = log2 - (1/N)*0.5*∑ywx + (1/N)*0.125*[∑(Wg*Xg)^2 + ∑(Wh*Xh)^2 + 2 * ∑(Wg*Xg * Wh*Xh)]
        """
        # mock data
        batch_loader = dataframe.DataLoader(
            train_data, ctx=ctx, batch_size=self.batch_size, mode="hetero", role="guest", sync_arbiter=True
        )
        # get encryptor
        ctx.arbiter("encryptor").get()

        w = tensor.tensor(torch.randn((train_data.num_features, 1), dtype=torch.float32))
        for i, iter_ctx in ctx.range(self.max_iter):
            logger.info(f"start iter {i}")
            j = 0
            for batch_ctx, (X, Y) in iter_ctx.iter(batch_loader):
                h = X.shape[0]

                # d
                Xw = tensor.matmul(X, w)
                d = 0.25 * Xw - 0.5 * Y
                loss = 0.125 / h * tensor.matmul(Xw.T, Xw) - 0.5 / h * tensor.matmul(Xw.T, Y)
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
        self.w = w

    def to_model(self):
        return {"w": self.w.to_local()._storage.data.tolist()}

    @classmethod
    def from_model(cls, model) -> "LrModuleGuest":
        ...
