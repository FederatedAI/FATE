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

import torch
from fate.arch import Context, tensor
from fate.arch.dataframe import DataLoader

from ..abc.module import HeteroModule

logger = logging.getLogger(__name__)


class LrModuleHost(HeteroModule):
    def __init__(
        self,
        max_iter,
        batch_size=None,
        learning_rate=0.01,
        alpha=1.0,
    ):
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.batch_size = batch_size

        self.w = None

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        batch_loader = DataLoader(train_data, ctx=ctx, batch_size=self.batch_size, mode="hetero", role="host")
        # get encryptor
        encryptor = ctx.arbiter("encryptor").get()

        w = tensor.tensor(torch.randn((train_data.num_features, 1), dtype=torch.float32))
        for i, iter_ctx in ctx.range(self.max_iter):
            logger.info(f"start iter {i}")
            j = 0
            for batch_ctx, X in iter_ctx.iter(batch_loader):
                h = X.shape[0]
                logger.info(f"start batch {j}")
                Xw_h = 0.25 * tensor.matmul(X, w)
                encryptor.encrypt(Xw_h).to(batch_ctx.guest, "Xw_h")
                encryptor.encrypt(tensor.matmul(Xw_h.T, Xw_h)).to(batch_ctx.guest, "Xw2_h")
                d = batch_ctx.guest.get("d")
                tensor.matmul(X.T, d).to(batch_ctx.arbiter, "g_enc")
                g = batch_ctx.arbiter.get("g")
                g = g / h + self.alpha * w
                w -= self.learning_rate * g
                logger.info(f"w={w}")
                j += 1

        self.w = w

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

    def predict(self, ctx, test_data):
        batch_loader = DataLoader(
            test_data,
            ctx=ctx,
            batch_size=-1,
            mode="hetero",
            role="host",
            sync_arbiter=False,
        )
        for X in batch_loader:
            output = tensor.matmul(X, self.w)
            print(output)

    @classmethod
    def from_model(cls, model) -> "LrModuleHost":
        lr = LrModuleHost(**model["metadata"])
        import torch

        lr.w = tensor.tensor(torch.tensor(model["w"]))
        return lr
