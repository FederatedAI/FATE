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

from fate.arch import Context
from fate.arch.dataframe import DataLoader
from fate.ml.abc.module import HeteroModule
from fate.ml.utils._model_param import initialize_param
from fate.ml.utils._optimizer import Optimizer, LRScheduler

logger = logging.getLogger(__name__)


class CoordinatedLinRModuleHost(HeteroModule):
    def __init__(
            self,
            max_iter,
            batch_size,
            optimizer_param,
            learning_rate_param,
            init_param
    ):
        self.max_iter = max_iter
        self.optimizer = Optimizer(optimizer_param["method"],
                                   optimizer_param["penalty"],
                                   optimizer_param["alpha"],
                                   optimizer_param["optimizer_params"])
        self.lr_scheduler = LRScheduler(learning_rate_param["method"],
                                        learning_rate_param["scheduler_params"])
        self.batch_size = batch_size
        self.init_param = init_param
        self.init_param["fit_intercept"] = False

        self.estimator = None

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        encryptor = ctx.arbiter("encryptor").get()
        estimator = CoordiantedLinREstimatorHost(max_iter=self.max_iter,
                                                 batch_size=self.batch_size,
                                                 optimizer=self.optimizer,
                                                 learning_rate_scheduler=self.lr_scheduler,
                                                 init_param=self.init_param)
        estimator.fit_model(ctx, encryptor, train_data, validate_data)
        self.estimator = estimator

    def predict(self, ctx, test_data):
        self.estimator.predict(test_data)

    def get_model(self):
        return {
            "estimator": self.estimator.get_model()
        }

    @classmethod
    def from_model(cls, model) -> "CoordinatedLinRModuleHost":
        linr = CoordinatedLinRModuleHost()
        estimator = CoordiantedLinREstimatorHost()
        estimator.restore(model["estimator"])
        linr.estimator = estimator

        return linr


class CoordiantedLinREstimatorHost(HeteroModule):
    def __init__(
            self,
            max_iter=None,
            batch_size=None,
            optimizer=None,
            learning_rate_scheduler=None,
            init_param=None
    ):
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.lr_scheduler = learning_rate_scheduler
        self.batch_size = batch_size
        self.init_param = init_param

        self.w = None
        self.start_iter = 0
        self.end_iter = -1
        self.is_converged = False

    def fit_model(self, ctx: Context, encryptor, train_data, validate_data=None) -> None:
        batch_loader = DataLoader(train_data, ctx=ctx, batch_size=self.batch_size, mode="hetero", role="host")

        coef_count = train_data.shape[1]
        w = self.w
        if self.w is None:
            w = initialize_param(coef_count, **self.init_param)
            self.optimizer.init_optimizer(model_parameter_length=w.size()[0])
            self.lr_scheduler.init_scheduler(optimizer=self.optimizer.optimizer)
        if self.end_iter >= 0:
            self.start_iter = self.end_iter + 1
        """for i, iter_ctx in ctx.range(self.start_iter, self.max_iter):"""
        # temp code start
        for i, iter_ctx in ctx.ctxs_range(self.max_iter):
            # temp code end
            logger.info(f"start iter {i}")
            j = 0
            self.optimizer.set_iters(i)
            for batch_ctx, X in iter_ctx.ctxs_zip(batch_loader):
                # h = X.shape[0]
                logger.info(f"start batch {j}")
                Xw_h = torch.matmul(X, w)
                encryptor.encrypt(Xw_h).to(batch_ctx.guest, "Xw_h")
                encryptor.encrypt(torch.matmul(Xw_h.T, Xw_h)).to(batch_ctx.guest, "Xw2_h")
                loss_norm = self.optimizer.loss_norm(w)
                if loss_norm is not None:
                    encryptor.encrypt(loss_norm).to(batch_ctx.guest, "h_loss")
                else:
                    batch_ctx.guest.put(h_loss=loss_norm)

                d = batch_ctx.guest.get("d")
                g = self.optimizer.add_regular_to_grad(torch.matmul(X.T, d), w, False)
                g.to(batch_ctx.arbiter, "g_enc")

                g = batch_ctx.arbiter.get("g")
                # g = g / h + self.alpha * w
                #  w -= self.learning_rate * g"
                w = self.optimizer.update_weights(w, g, False, self.lr_scheduler.lr)
                logger.info(f"w={w}")
                j += 1
            self.is_converged = ctx.arbiter("converge_flag").get()
            if self.is_converged:
                self.end_iter = i
                break
            self.lr_scheduler.step()
        if not self.is_converged:
            self.end_iter = self.max_iter
        self.w = w
        logger.debug(f"Finish training at {self.end_iter}th iteration.")

    def predict(self, ctx, test_data):
        X = test_data.values.as_tensor()
        output = torch.matmul(X, self.w)
        ctx.guest.put("h_pred", output)

    def get_model(self):
        return {
            "w": self.w.tolist(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "end_iter": self.end_iter,
            "converged": self.is_converged
        }

    def restore(self, model):
        self.w = torch.tensor(model["w"])
        self.optimizer.load_state_dict(model["optimizer"])
        self.lr_scheduler.load_state_dict(model["lr_scheduler"])
        self.end_iter = model["end_iter"]
        self.is_converged = model["is_converged"]
