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
import logging

from fate.arch import Context, dataframe, tensor

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
        # # get encryptor
        # ctx.arbiter("encryptor").get()

        w = tensor.randn((train_data.num_features, 1), dtype=dtype.float32)
        for i, iter_ctx in ctx.on_iterations.ctxs_range(self.max_iter):
            logger.info(f"start iter {i}")
            j = 0
            for batch_ctx, (X, Y) in iter_ctx.on_batches.ctxs_zip(batch_loader):
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

                # gradient
                batch_ctx.arbiter.put("g_enc", X.T @ d)
                g: tensor.Tensor = batch_ctx.arbiter.get("g")
                # apply l2 penalty
                g = g / h + self.alpha * w
                w -= self.learning_rate * g
                logger.info(f"w={w}")
                j += 1
        self.w = w

    def predict(self, ctx, test_data):
        batch_loader = dataframe.DataLoader(
            test_data,
            ctx=ctx,
            batch_size=-1,
            mode="hetero",
            role="guest",
            sync_arbiter=False,
        )
        for X, y in batch_loader:
            output = tensor.matmul(X, self.w)

        return output

    def get_model(self):
        return {
            "w": self.w.to_local()._storage.data.tolist(),
            "metadata": {
                "max_iter": self.max_iter,
                "batch_size": self.batch_size,
                "learning_rate": self.learning_rate,
                "alpha": self.alpha,
            },
        }

    @classmethod
    def from_model(cls, model) -> "LrModuleGuest":
        lr = LrModuleGuest(**model["metadata"])
        import torch

        lr.w = tensor.tensor(torch.tensor(model["w"]))
        return lr
