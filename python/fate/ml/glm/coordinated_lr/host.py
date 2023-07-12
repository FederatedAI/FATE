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
from fate.ml.utils._optimizer import LRScheduler, Optimizer

logger = logging.getLogger(__name__)


class CoordinatedLRModuleHost(HeteroModule):
    def __init__(
            self, epochs=None, batch_size=None, optimizer_param=None, learning_rate_param=None, init_param=None
    ):
        self.epochs = epochs
        self.learning_rate_param = learning_rate_param
        self.optimizer_param = optimizer_param
        self.batch_size = batch_size
        self.init_param = init_param

        # host never has fit intercept
        self.init_param["fit_intercept"] = False

        self.estimator = None
        self.ovr = False
        self.label_count = False

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        encryptor = ctx.arbiter("encryptor").get()
        self.label_count = ctx.guest("label_count").get()
        if self.label_count > 2:
            self.ovr = True
            self.estimator = {}
            for i, class_ctx in ctx.sub_ctx("class").ctxs_range(self.label_count):
                # optimizer = copy.deepcopy(self.optimizer)
                # lr_scheduler = copy.deepcopy(self.lr_scheduler)
                optimizer = Optimizer(
                    self.optimizer_param["method"],
                    self.optimizer_param["penalty"],
                    self.optimizer_param["alpha"],
                    self.optimizer_param["optimizer_params"],
                )
                lr_scheduler = LRScheduler(self.learning_rate_param["method"],
                                           self.learning_rate_param["scheduler_params"])
                single_estimator = CoordinatedLREstimatorHost(
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    optimizer=optimizer,
                    learning_rate_scheduler=lr_scheduler,
                    init_param=self.init_param,
                )
                single_estimator.fit_single_model(class_ctx, encryptor, train_data, validate_data)
                self.estimator[i] = single_estimator
        else:
            optimizer = Optimizer(
                self.optimizer_param["method"],
                self.optimizer_param["penalty"],
                self.optimizer_param["alpha"],
                self.optimizer_param["optimizer_params"],
            )
            lr_scheduler = LRScheduler(self.learning_rate_param["method"],
                                       self.learning_rate_param["scheduler_params"])
            single_estimator = CoordinatedLREstimatorHost(
                epochs=self.epochs,
                batch_size=self.batch_size,
                optimizer=optimizer,
                learning_rate_scheduler=lr_scheduler,
                init_param=self.init_param,
            )
            single_estimator.fit_single_model(ctx, encryptor, train_data, validate_data)
            self.estimator = single_estimator

    def predict(self, ctx, test_data):
        if self.ovr:
            for i, class_ctx in ctx.sub_ctx("class").ctxs_range(self.label_count):
                estimator = self.estimator[i]
                estimator.predict(class_ctx, test_data)
        else:
            self.estimator.predict(ctx, test_data)

    def get_model(self):
        all_estimator = {}
        if self.ovr:
            for label_idx, estimator in self.estimator.items():
                all_estimator[label_idx] = estimator.get_model()
        else:
            all_estimator = self.estimator.get_model()
        return {"data": {"estimator": all_estimator},
                "meta": {"label_count": self.label_count,
                         "ovr": self.ovr,
                         "epochs": self.epochs,
                         "batch_size": self.batch_size,
                         "learning_rate_param": self.learning_rate_param,
                         "optimizer_param": self.optimizer_param,
                         "init_param": self.init_param},
                }

    @classmethod
    def from_model(cls, model) -> "CoordinatedLRModuleHost":
        lr = CoordinatedLRModuleHost(epochs=model["meta"]["epochs"],
                                     batch_size=model["meta"]["batch_size"],
                                     learning_rate_param=model["meta"]["learning_rate_param"],
                                     optimizer_param=model["meta"]["optimizer_param"],
                                     init_param=model["meta"]["init_param"])
        lr.label_count = model["meta"]["label_count"]
        lr.ovr = model["meta"]["ovr"]

        all_estimator = model["data"]["estimator"]
        if lr.ovr:
            lr.estimator = {label: CoordinatedLREstimatorHost().restore(d) for label, d in all_estimator.items()}
        else:
            estimator = CoordinatedLREstimatorHost()
            estimator.restore(all_estimator)
            lr.estimator = estimator
        logger.info(f"finish from model")

        return lr


class CoordinatedLREstimatorHost(HeteroModule):
    def __init__(self, epochs=None, batch_size=None, optimizer=None, learning_rate_scheduler=None, init_param=None):
        self.epochs = epochs
        self.optimizer = optimizer
        self.lr_scheduler = learning_rate_scheduler
        self.batch_size = batch_size
        self.init_param = init_param

        self.w = None
        self.start_epoch = 0
        self.end_epoch = -1
        self.is_converged = False

    def fit_single_model(self, ctx: Context, encryptor, train_data, validate_data=None) -> None:
        coef_count = train_data.shape[1]
        w = self.w
        if self.w is None:
            w = initialize_param(coef_count, **self.init_param)
            self.optimizer.init_optimizer(model_parameter_length=w.size()[0])
            self.lr_scheduler.init_scheduler(optimizer=self.optimizer.optimizer)
        batch_loader = DataLoader(train_data, ctx=ctx, batch_size=self.batch_size, mode="hetero", role="host")
        if self.end_epoch >= 0:
            self.start_epoch = self.end_epoch + 1
        for i, iter_ctx in ctx.on_iterations.ctxs_range(self.start_epoch, self.epochs):
            self.optimizer.set_iters(i)
            logger.info(f"self.optimizer set epoch{i}")
            for batch_ctx, X in iter_ctx.on_batches.ctxs_zip(batch_loader):
                h = X.shape[0]
                Xw_h = 0.25 * torch.matmul(X, w.detach())
                batch_ctx.guest.put("Xw_h", encryptor.encrypt(Xw_h))
                batch_ctx.guest.put("Xw2_h", encryptor.encrypt(torch.matmul(Xw_h.T, Xw_h)))

                loss_norm = self.optimizer.loss_norm(w)
                if loss_norm is not None:
                    batch_ctx.guest.put("h_loss", encryptor.encrypt(loss_norm))
                else:
                    batch_ctx.guest.put(h_loss=loss_norm)

                d = batch_ctx.guest.get("d")
                g = 1 / h * torch.matmul(X.T, d)
                g = self.optimizer.add_regular_to_grad(g, w, False)
                batch_ctx.arbiter.put("g_enc", g)
                g = batch_ctx.arbiter.get("g")

                w = self.optimizer.update_weights(w, g, False, self.lr_scheduler.lr)
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
        X = test_data.values.as_tensor()
        output = torch.matmul(X, self.w)
        ctx.guest.put("h_pred", output)

    def get_model(self):
        return {
            "w": self.w.tolist(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "end_epoch": self.end_epoch,
            "is_converged": self.is_converged,
        }

    def restore(self, model):
        self.w = torch.tensor(model["w"])
        self.optimizer = Optimizer()
        self.lr_scheduler = LRScheduler()
        self.optimizer.load_state_dict(model["optimizer"])
        self.lr_scheduler.load_state_dict(model["lr_scheduler"], self.optimizer.optimizer)
        self.end_epoch = model["end_epoch"]
        self.is_converged = model["is_converged"]
