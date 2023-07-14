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
from fate.ml.utils._model_param import initialize_param, serialize_param, deserialize_param
from fate.ml.utils._optimizer import Optimizer, LRScheduler

logger = logging.getLogger(__name__)


class CoordinatedLinRModuleGuest(HeteroModule):
    def __init__(
            self,
            epochs=None,
            batch_size=None,
            optimizer_param=None,
            learning_rate_param=None,
            init_param=None
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_param = optimizer_param
        self.learning_rate_param = learning_rate_param
        self.init_param = init_param

        self.estimator = None

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.estimator.batch_size = batch_size

    def set_epochs(self, epochs):
        self.epochs = epochs
        self.estimator.epochs = epochs

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        if self.estimator is None:
            optimizer = Optimizer(
                self.optimizer_param["method"],
                self.optimizer_param["penalty"],
                self.optimizer_param["alpha"],
                self.optimizer_param["optimizer_params"],
            )
            lr_scheduler = LRScheduler(self.learning_rate_param["method"],
                                       self.learning_rate_param["scheduler_params"])
            estimator = CoordinatedLinREstimatorGuest(epochs=self.epochs,
                                                      batch_size=self.batch_size,
                                                      optimizer=optimizer,
                                                      learning_rate_scheduler=lr_scheduler,
                                                      init_param=self.init_param)
            self.estimator = estimator
        self.estimator.fit_model(ctx, train_data, validate_data)

    def predict(self, ctx, test_data):
        prob = self.estimator.predict(ctx, test_data)
        return prob

    def get_model(self):
        return {"data": {"estimator": self.estimator.get_model()},
                "meta": {"epochs": self.epochs,
                         "batch_size": self.batch_size,
                         "learning_rate_param": self.learning_rate_param,
                         "init_param": self.init_param,
                         "optimizer_param": self.optimizer_param}
                }

    @classmethod
    def from_model(cls, model) -> "CoordinatedLinRModuleGuest":
        linr = CoordinatedLinRModuleGuest(optimizer_param=model["meta"]["optimizer_param"],
                                          learning_rate_param=model["meta"]["learning_rate_param"],
                                          batch_size=model["meta"]["batch_size"],
                                          init_param=model["meta"]["init_param"])
        estimator = CoordinatedLinREstimatorGuest(epochs=model["meta"]["epochs"],
                                                  batch_size=model["meta"]["batch_size"],
                                                  init_param=model["meta"]["init_param"])
        estimator.restore(model["data"]["estimator"])
        linr.estimator = estimator

        return linr


class CoordinatedLinREstimatorGuest(HeteroModule):
    def __init__(
            self,
            epochs=None,
            batch_size=None,
            optimizer=None,
            learning_rate_scheduler=None,
            init_param=None
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr_scheduler = learning_rate_scheduler
        self.init_param = init_param

        self.w = None
        self.start_epoch = 0
        self.end_epoch = -1
        self.is_converged = False

    def fit_model(self, ctx, train_data, validate_data=None):
        coef_count = train_data.shape[1]
        logger.debug(f"init param: {self.init_param}")
        if self.init_param.get("fit_intercept"):
            logger.debug(f"add intercept to train data")
            train_data["intercept"] = 1.0
        w = self.w
        if self.w is None:
            w = initialize_param(coef_count, **self.init_param)
            self.optimizer.init_optimizer(model_parameter_length=w.size()[0])
            self.lr_scheduler.init_scheduler(optimizer=self.optimizer.optimizer)
        batch_loader = dataframe.DataLoader(
            train_data, ctx=ctx, batch_size=self.batch_size, mode="hetero", role="guest", sync_arbiter=True
        )
        # if self.end_epoch >= 0:
        #    self.start_epoch = self.end_epoch + 1

        for i, iter_ctx in ctx.on_iterations.ctxs_range(self.epochs):
            self.optimizer.set_iters(i)
            logger.info(f"self.optimizer set epoch {i}")
            for batch_ctx, batch_data in iter_ctx.on_batches.ctxs_zip(batch_loader):
                X = batch_data.x
                Y = batch_data.label
                weight = batch_data.weight
                h = X.shape[0]
                Xw = torch.matmul(X, w.detach())
                d = Xw - Y
                loss = 0.5 / h * torch.matmul(d.T, d)
                if self.optimizer.l1_penalty or self.optimizer.l2_penalty:
                    loss_norm = self.optimizer.loss_norm(w)
                    loss += loss_norm
                Xw_h_all = batch_ctx.hosts.get("Xw_h")
                for Xw_h in Xw_h_all:
                    d += Xw_h
                    loss += 1 / h * torch.matmul(Xw.T, Xw_h)

                if weight:
                    d = d * weight
                batch_ctx.hosts.put(d=d)

                for Xw2_h in batch_ctx.hosts.get("Xw2_h"):
                    loss += 0.5 / h * Xw2_h
                h_loss_list = batch_ctx.hosts.get("h_loss")
                for h_loss in h_loss_list:
                    if h_loss is not None:
                        loss += h_loss

                if len(Xw_h_all) == 1:
                    batch_ctx.arbiter.put(loss=loss)

                # gradient
                g = 1 / h * torch.matmul(X.T, d)
                g = self.optimizer.add_regular_to_grad(g, w, self.init_param.get("fit_intercept"))
                batch_ctx.arbiter.put("g_enc", g)
                g = batch_ctx.arbiter.get("g")

                w = self.optimizer.update_weights(w, g, self.init_param.get("fit_intercept"), self.lr_scheduler.lr)
                # logger.info(f"w={w}")
            self.is_converged = iter_ctx.arbiter("converge_flag").get()
            if self.is_converged:
                self.end_epoch = i
                break
            if i < self.epochs - 1:
                self.lr_scheduler.step()
        if not self.is_converged:
            self.end_epoch = self.epochs
        self.w = w
        logger.debug(f"Finish training at {self.end_epoch}th epoch.")

    def predict(self, ctx, test_data):
        if self.init_param.get("fit_intercept"):
            test_data["intercept"] = 1.0
        X = test_data.values.as_tensor()
        pred = torch.matmul(X, self.w)
        for h_pred in ctx.hosts.get("h_pred"):
            pred += h_pred
        return pred

    def get_model(self):
        """w = self.w.tolist()
        intercept = None
        if self.init_param.get("fit_intercept"):
            w = w[:-1]
            intercept = w[-1]"""
        param = serialize_param(self.w, self.init_param.get("fit_intercept"))
        return {
            "param": param,
            # "intercept": intercept,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "end_epoch": self.end_epoch,
            "is_converged": self.is_converged,
            "fit_intercept": self.init_param.get("fit_intercept")
        }

    def restore(self, model):
        """w = model["w"]
        if model["fit_intercept"]:
            w.append(model["intercept"])
        self.w = torch.tensor(w)
        """
        self.w = deserialize_param(model["param"], model["fit_intercept"])
        self.optimizer = Optimizer()
        self.lr_scheduler = LRScheduler()
        self.optimizer.load_state_dict(model["optimizer"])
        self.lr_scheduler.load_state_dict(model["lr_scheduler"], self.optimizer.optimizer)
        self.end_epoch = model["end_epoch"]
        self.is_converged = model["is_converged"]
