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

import copy
import logging

import torch

from fate.arch import Context, dataframe
from fate.ml.abc.module import HeteroModule
from fate.ml.utils._model_param import initialize_param
from fate.ml.utils._optimizer import LRScheduler, Optimizer

logger = logging.getLogger(__name__)


class CoordinatedLRModuleGuest(HeteroModule):
    def __init__(
        self,
        max_iter=None,
        batch_size=None,
        optimizer_param=None,
        learning_rate_param=None,
        init_param=None,
        threshold=0.5,
    ):
        self.max_iter = max_iter
        self.batch_size = batch_size

        self.optimizer = Optimizer(
            optimizer_param["method"],
            optimizer_param["penalty"],
            optimizer_param["alpha"],
            optimizer_param["optimizer_params"],
        )
        self.lr_scheduler = LRScheduler(learning_rate_param["method"], learning_rate_param["scheduler_params"])

        # temp code block start
        """self.optimizer = Optimizer(
            optimizer_param.method, optimizer_param.penalty, optimizer_param.alpha, optimizer_param.optimizer_params
        )
        self.lr_scheduler = LRScheduler(learning_rate_param.method, learning_rate_param.scheduler_params)"""
        # temp ode block ends
        self.init_param = init_param
        self.threshold = threshold

        self.estimator = None
        self.ovr = False
        self.labels = []

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        # encryptor = ctx.arbiter("encryptor").get()
        original_label = train_data.label
        train_data_binarized_label = train_data.label.get_dummies()
        label_count = train_data_binarized_label.shape[1]
        ctx.arbiter.put("label_count", label_count)
        ctx.hosts.put("label_count", label_count)
        self.labels = [label_name.split("_")[1] for label_name in train_data_binarized_label.columns]
        with_weight = train_data.weight is not None
        """
        # temp code start
        label_count = 2
        ctx.arbiter.put("label_count", label_count)
        ctx.hosts.put("label_count", label_count)
        with_weight = False
        # temp code end
        """
        if label_count > 2:
            self.ovr = True
            self.estimator = {}
            # train_data_binarized_label = train_data.label.one_hot()
            for i, class_ctx in ctx.sub_ctx("class").ctxs_range(label_count):
                optimizer = copy.deepcopy(self.optimizer)
                single_estimator = CoordinatedLREstimatorGuest(
                    max_iter=self.max_iter,
                    batch_size=self.batch_size,
                    optimizer=optimizer,
                    learning_rate_scheduler=self.lr_scheduler,
                    init_param=self.init_param,
                )
                train_data.label = train_data_binarized_label[f"{train_data.schema.label_name}_{self.labels[i]}"]
                single_estimator.fit_single_model(class_ctx, train_data, validate_data, with_weight=with_weight)
                self.estimator[i] = single_estimator
            train_data.label = original_label
        else:
            single_estimator = CoordinatedLREstimatorGuest(
                max_iter=self.max_iter,
                batch_size=self.batch_size,
                optimizer=self.optimizer,
                learning_rate_scheduler=self.lr_scheduler,
                init_param=self.init_param,
            )
            single_estimator.fit_single_model(ctx, train_data, validate_data, with_weight=with_weight)
            self.estimator = single_estimator

    def predict(self, ctx, test_data):
        if self.ovr:
            predict_score = test_data.create_dataframe(with_label=False, with_weight=False)
            for i, class_ctx in ctx.ctxs_range(len(self.labels)):
                estimator = self.estimator[i]
                pred = estimator.predict(ctx, test_data)
                predict_score[self.labels[i]] = pred
        else:
            predict_score = self.estimator.predict(ctx, test_data)
        return predict_score

    def get_model(self):
        all_estimator = {}
        if self.ovr:
            for label, estimator in self.estimator.items():
                all_estimator[label] = estimator.get_model()
        else:
            all_estimator = self.estimator.get_model()
        return {"estimator": all_estimator, "labels": self.labels, "ovr": self.ovr, "threshold": self.threshold}

    @classmethod
    def from_model(cls, model) -> "CoordinatedLRModuleGuest":
        lr = CoordinatedLRModuleGuest()
        lr.ovr = model["ovr"]
        lr.labels = model["labels"]
        lr.threshold = model["threshold"]

        all_estimator = model["estimator"]
        if lr.ovr:
            lr.estimator = {label: CoordinatedLREstimatorGuest().restore(d) for label, d in all_estimator.items()}
        else:
            estimator = CoordinatedLREstimatorGuest()
            estimator.restore(all_estimator)
            lr.estimator = estimator

        return lr


class CoordinatedLREstimatorGuest(HeteroModule):
    def __init__(self, max_iter=None, batch_size=None, optimizer=None, learning_rate_scheduler=None, init_param=None):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr_scheduler = learning_rate_scheduler
        self.init_param = init_param

        self.w = None
        self.start_iter = 0
        self.end_iter = -1
        self.is_converged = False
        self.with_weight = False

    def fit_single_model(self, ctx: Context, train_data, validate_data=None, with_weight=False):
        """
        l(w) = 1/h * Σ(log(2) - 0.5 * y * xw + 0.125 * (wx)^2)
        ∇l(w) = 1/h * Σ(0.25 * xw - 0.5 * y)x = 1/h * Σdx
        where d = 0.25(xw - 2y)
        loss = log2 - (1/N)*0.5*∑ywx + (1/N)*0.125*[∑(Wg*Xg)^2 + ∑(Wh*Xh)^2 + 2 * ∑(Wg*Xg * Wh*Xh)]
        """
        coef_count = train_data.shape[1]
        # @todo: need to make sure add single-valued column works
        if self.init_param.get("fit_intercept"):
            train_data["intercept"] = 1.0
            coef_count += 1

        # temp code start
        # coef_count = 10
        # temp code end

        w = self.w
        if w is None:
            w = initialize_param(coef_count, **self.init_param)
            self.optimizer.init_optimizer(model_parameter_length=w.size()[0])
            self.lr_scheduler.init_scheduler(optimizer=self.optimizer.optimizer)
            # temp code end

        batch_loader = dataframe.DataLoader(
            train_data, ctx=ctx, batch_size=self.batch_size, mode="hetero", role="guest", sync_arbiter=True
        )
        if self.end_iter >= 0:
            self.start_iter = self.end_iter + 1
        """if train_data.weight:
            self.with_weight = True"""

        # for i, iter_ctx in ctx.ctxs_range(self.start_iter, self.max_iter):
        # temp code start
        for i, iter_ctx in ctx.on_iterations.ctxs_range(self.max_iter):
            # temp code end
            # logger.info(f"start iter {i}")
            j = 0
            self.optimizer.set_iters(i)
            logger.info(f"self.optimizer set iters {i}")
            # todo: if self.with_weight: include weight in batch result
            for batch_ctx, (X, Y) in iter_ctx.on_batches.ctxs_zip(batch_loader):
                h = X.shape[0]

                Xw = torch.matmul(X, w.detach())
                d = 0.25 * Xw - 0.5 * Y
                loss = 0.125 / h * torch.matmul(Xw.T, Xw) - 0.5 / h * torch.matmul(Xw.T, Y)

                if self.optimizer.l1_penalty or self.optimizer.l2_penalty:
                    loss_norm = self.optimizer.loss_norm(w)
                    loss += loss_norm
                for Xw_h in batch_ctx.hosts.get("Xw_h"):
                    d += Xw_h
                    loss -= 0.5 / h * torch.matmul(Y.T, Xw_h)
                    loss += 0.25 / h * torch.matmul(Xw.T, Xw_h)
                # if with_weight:
                #    d = d * weight
                for Xw2_h in batch_ctx.hosts.get("Xw2_h"):
                    loss += 0.125 / h * Xw2_h

                batch_ctx.hosts.put(d=d)
                h_loss_list = batch_ctx.hosts.get("h_loss")
                for h_loss in h_loss_list:
                    if h_loss is not None:
                        loss += h_loss
                batch_ctx.arbiter.put(loss=loss)

                # gradient
                g = X.T @ d
                batch_ctx.arbiter.put("g_enc", g)
                g = batch_ctx.arbiter.get("g")
                g = self.optimizer.add_regular_to_grad(g, w, self.init_param.get("fit_intercept"))
                # self.optimizer.step(g)
                w = self.optimizer.update_weights(w, g, self.init_param.get("fit_intercept"), self.lr_scheduler.lr)

                # logger.info(f"w={w}")
                j += 1
            self.is_converged = iter_ctx.arbiter("converge_flag").get()
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
        pred = torch.sigmoid(pred)
        return pred

    def get_model(self):
        return {
            "w": self.w.tolist(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "end_iter": self.end_iter,
            "converged": self.is_converged,
        }

    def restore(self, model):
        self.w = torch.tensor(model["w"])
        self.optimizer.load_state_dict(model["optimizer"])
        self.lr_scheduler.load_state_dict(model["lr_scheduler"])
        self.end_iter = model["end_iter"]
        self.is_converged = model["is_converged"]
