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

from fate.arch import dataframe, Context
from fate.ml.abc.module import HeteroModule
from fate.ml.utils._model_param import initialize_param
from fate.ml.utils._optimizer import Optimizer, LRScheduler

logger = logging.getLogger(__name__)


class CoordinatedLinRModuleGuest(HeteroModule):
    def __init__(
            self,
            max_iter=None,
            batch_size=None,
            optimizer_param=None,
            learning_rate_param=None,
            init_param=None
    ):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.optimizer = Optimizer(optimizer_param["method"],
                                   optimizer_param["penalty"],
                                   optimizer_param["alpha"],
                                   optimizer_param["optimizer_params"])
        self.lr_scheduler = LRScheduler(learning_rate_param["method"],
                                        learning_rate_param["scheduler_params"])

        self.init_param = init_param

        self.estimator = None

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        with_weight = train_data.weight is not None

        estimator = CoordinatedLinREstimatorGuest(max_iter=self.max_iter,
                                                  batch_size=self.batch_size,
                                                  optimizer=self.optimizer,
                                                  learning_rate_scheduler=self.lr_scheduler,
                                                  init_param=self.init_param)
        estimator.fit_model(ctx, train_data, validate_data, with_weight=with_weight)
        self.estimator = estimator

    def predict(self, ctx, test_data):
        prob = self.estimator.predict(ctx, test_data)
        return prob

    def get_model(self):
        return {
            "estimator": self.estimator.get_model(),
        }

    @classmethod
    def from_model(cls, model) -> "CoordinatedLinRModuleGuest":
        linr = CoordinatedLinRModuleGuest()
        estimator = CoordinatedLinREstimatorGuest()
        estimator.restore(model["estimator"])
        linr.estimator = estimator

        return linr


class CoordinatedLinREstimatorGuest(HeteroModule):
    def __init__(
            self,
            max_iter=None,
            batch_size=None,
            optimizer=None,
            learning_rate_scheduler=None,
            init_param=None
    ):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr_scheduler = learning_rate_scheduler
        self.init_param = init_param

        self.w = None
        self.start_iter = 0
        self.end_iter = -1
        self.is_converged = False

    def fit_model(self, ctx, train_data, validate_data=None, with_weight=False):
        coef_count = train_data.shape[1]
        if self.init_param.get("fit_intercept"):
            train_data["intercept"] = 1
            coef_count += 1
        w = self.w
        if self.w is None:
            w = initialize_param(coef_count, **self.init_param)
            self.optimizer.init_optimizer(model_parameter_length=w.size()[0])
            self.lr_scheduler.init_scheduler(optimizer=self.optimizer.optimizer)
        batch_loader = dataframe.DataLoader(
            train_data, ctx=ctx, batch_size=self.batch_size, mode="hetero", role="guest", sync_arbiter=True,
            # with_weight=True
        )
        if self.end_iter >= 0:
            self.start_iter = self.end_iter + 1

        # for i, iter_ctx in ctx.ctxs_range(self.start_iter, self.max_iter):
        # temp code start
        for i, iter_ctx in ctx.ctxs_range(self.max_iter):
            # temp code end
            logger.info(f"start iter {i}")
            j = 0
            self.optimizer.set_iters(i)
            # for batch_ctx, (X, Y, weight) in iter_ctx.iter(batch_loader):
            for batch_ctx, (X, Y) in iter_ctx.ctxs_zip(batch_loader):
                h = X.shape[0]
                Xw = torch.matmul(X, w)
                d = Xw - Y
                loss = 1 / 2 / h * torch.matmul(d.T, d)
                if self.optimizer.l1_penalty or self.optimizer.l2_penalty:
                    loss_norm = self.optimizer.loss_norm(w)
                    loss += loss_norm
                for Xw_h in batch_ctx.hosts.get("Xw_h"):
                    d += Xw_h
                    loss += 1 / h * torch.matmul(Xw.T, Xw_h)
                # if with_weight:
                #    d = d * weight
                for Xw2_h in batch_ctx.hosts.get("Xw2_h"):
                    loss += 1 / 2 / h * Xw2_h

                batch_ctx.hosts.put(d=d)
                h_loss_list = batch_ctx.hosts.get("h_loss")
                for h_loss in h_loss_list:
                    if h_loss is not None:
                        loss += h_loss
                batch_ctx.arbiter.put(loss=loss)

                # gradient
                g = X.T @ d
                batch_ctx.arbiter.put("g_enc", X.T @ g)
                g = batch_ctx.arbiter.get("g")

                g = self.optimizer.add_regular_to_grad(g, w, self.init_param.get("fit_intercept"))
                w = self.optimizer.update_weights(w, g, self.init_param.get("fit_intercept"), self.lr_scheduler.lr)
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

    def predict(self, ctx, test_data):
        X = test_data.values.as_tensor()
        pred = torch.matmul(X, self.w)
        for h_pred in ctx.hosts.get("h_pred"):
            pred += h_pred
        return pred

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
