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

import numpy as np
import torch

from fate.arch import Context, dataframe
from fate.ml.abc.module import HeteroModule
from fate.ml.utils import predict_tools
from fate.ml.utils._model_param import (
    check_overflow,
    deserialize_param,
    initialize_param,
    serialize_param,
)
from fate.ml.utils._optimizer import LRScheduler, Optimizer

logger = logging.getLogger(__name__)


class CoordinatedLRModuleGuest(HeteroModule):
    def __init__(
        self,
        epochs=None,
        batch_size=None,
        optimizer_param=None,
        learning_rate_param=None,
        init_param=None,
        threshold=0.5,
            floating_point_precision=23
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate_param = learning_rate_param
        self.optimizer_param = optimizer_param
        self.init_param = init_param
        self.threshold = threshold
        self.floating_point_precision = floating_point_precision

        self.estimator = None
        self.ovr = False
        self.labels = None

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        if self.ovr:
            for estimator in self.estimator.values():
                estimator.batch_size = batch_size
        else:
            self.estimator.batch_size = batch_size

    def set_epochs(self, epochs):
        self.epochs = epochs
        if self.ovr:
            for estimator in self.estimator.values():
                estimator.epochs = epochs
        else:
            self.estimator.epochs = epochs

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        # original_label = train_data.label
        train_data_binarized_label = train_data.label.get_dummies()
        label_count = train_data_binarized_label.shape[1]
        ctx.arbiter.put("label_count", label_count)
        ctx.hosts.put("label_count", label_count)
        encryptor = ctx.arbiter("encryptor").get()
        labels = [int(label_name.split("_")[1]) for label_name in train_data_binarized_label.columns]
        if self.labels is None:
            self.labels = sorted(labels)
        if label_count > 2 or self.ovr:
            logger.info(f"OVR data provided, will train OVR models.")
            self.ovr = True
            warm_start = True
            if self.estimator is None:
                self.estimator = {}
                warm_start = False
            for i, class_ctx in ctx.sub_ctx("class").ctxs_range(label_count):
                logger.info(f"start train for {i}th class")
                # optimizer = copy.deepcopy(self.optimizer)
                if not warm_start:
                    optimizer = Optimizer(
                        self.optimizer_param["method"],
                        self.optimizer_param["penalty"],
                        self.optimizer_param["alpha"],
                        self.optimizer_param["optimizer_params"],
                    )
                    lr_scheduler = LRScheduler(
                        self.learning_rate_param["method"], self.learning_rate_param["scheduler_params"]
                    )
                    single_estimator = CoordinatedLREstimatorGuest(
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        optimizer=optimizer,
                        learning_rate_scheduler=lr_scheduler,
                        init_param=self.init_param,
                        floating_point_precision=self.floating_point_precision
                    )
                else:
                    # warm start
                    logger.info("estimator is not none, will train with warm start")
                    # single_estimator = self.estimator[self.labels.index(labels[i])]
                    single_estimator = self.estimator[i]
                    single_estimator.epochs = self.epochs
                    single_estimator.batch_size = self.batch_size
                class_train_data = train_data.copy()
                class_validate_data = validate_data
                if validate_data:
                    class_validate_data = validate_data.copy()
                class_train_data.label = train_data_binarized_label[train_data_binarized_label.columns[i]]
                single_estimator.fit_single_model(class_ctx, encryptor, class_train_data, class_validate_data)
                self.estimator[i] = single_estimator

        else:
            if self.estimator is None:
                optimizer = Optimizer(
                    self.optimizer_param["method"],
                    self.optimizer_param["penalty"],
                    self.optimizer_param["alpha"],
                    self.optimizer_param["optimizer_params"],
                )
                lr_scheduler = LRScheduler(
                    self.learning_rate_param["method"], self.learning_rate_param["scheduler_params"]
                )
                single_estimator = CoordinatedLREstimatorGuest(
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    optimizer=optimizer,
                    learning_rate_scheduler=lr_scheduler,
                    init_param=self.init_param,
                    floating_point_precision=self.floating_point_precision
                )
            else:
                logger.info("estimator is not none, will train with warm start")
                single_estimator = self.estimator
                single_estimator.epochs = self.epochs
                single_estimator.batch_size = self.batch_size
            train_data_fit = train_data.copy()
            validate_data_fit = validate_data
            if validate_data:
                validate_data_fit = validate_data.copy()
            single_estimator.fit_single_model(ctx, encryptor, train_data_fit, validate_data_fit)
            self.estimator = single_estimator

    def predict(self, ctx, test_data):
        pred_df = test_data.create_frame(with_label=True, with_weight=False)
        if self.ovr:
            pred_score = test_data.create_frame(with_label=False, with_weight=False)
            for i, class_ctx in ctx.sub_ctx("class").ctxs_range(len(self.labels)):
                estimator = self.estimator[i]
                pred = estimator.predict(class_ctx, test_data)
                pred_score[str(self.labels[i])] = pred
            pred_df[predict_tools.PREDICT_SCORE] = pred_score.apply_row(lambda v: [list(v)])
            predict_result = predict_tools.compute_predict_details(
                pred_df, task_type=predict_tools.MULTI, classes=self.labels
            )
        else:
            predict_score = self.estimator.predict(ctx, test_data)
            pred_df[predict_tools.PREDICT_SCORE] = predict_score
            predict_result = predict_tools.compute_predict_details(
                pred_df, task_type=predict_tools.BINARY, classes=self.labels, threshold=self.threshold
            )

        return predict_result

    def get_model(self):
        all_estimator = {}
        if self.ovr:
            for label, estimator in self.estimator.items():
                all_estimator[label] = estimator.get_model()
        else:
            all_estimator = self.estimator.get_model()
        return {
            "data": {"estimator": all_estimator},
            "meta": {
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "learning_rate_param": self.learning_rate_param,
                "init_param": self.init_param,
                "optimizer_param": self.optimizer_param,
                "labels": self.labels,
                "ovr": self.ovr,
                "threshold": self.threshold,
                "floating_point_precision": self.floating_point_precision,
            },
        }

    @classmethod
    def from_model(cls, model) -> "CoordinatedLRModuleGuest":
        lr = CoordinatedLRModuleGuest(
            epochs=model["meta"]["epochs"],
            batch_size=model["meta"]["batch_size"],
            learning_rate_param=model["meta"]["learning_rate_param"],
            optimizer_param=model["meta"]["optimizer_param"],
            threshold=model["meta"]["threshold"],
            init_param=model["meta"]["init_param"],
            floating_point_precision=model["meta"]["floating_point_precision"],
        )
        lr.ovr = model["meta"]["ovr"]
        lr.labels = model["meta"]["labels"]

        all_estimator = model["data"]["estimator"]
        lr.estimator = {}
        if lr.ovr:
            for label, d in all_estimator.items():
                estimator = CoordinatedLREstimatorGuest(
                    epochs=model["meta"]["epochs"],
                    batch_size=model["meta"]["batch_size"],
                    init_param=model["meta"]["init_param"],
                    floating_point_precision=model["meta"]["floating_point_precision"],
                )
                estimator.restore(d)
                lr.estimator[int(label)] = estimator
        else:
            estimator = CoordinatedLREstimatorGuest(
                epochs=model["meta"]["epochs"],
                batch_size=model["meta"]["batch_size"],
                init_param=model["meta"]["init_param"],
                floating_point_precision=model["meta"]["floating_point_precision"],
            )
            estimator.restore(all_estimator)
            lr.estimator = estimator

        return lr


class CoordinatedLREstimatorGuest(HeteroModule):
    def __init__(self, epochs=None, batch_size=None, optimizer=None, learning_rate_scheduler=None, init_param=None,
                 floating_point_precision=23):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr_scheduler = learning_rate_scheduler
        self.init_param = init_param
        self.floating_point_precision = floating_point_precision
        self._fixpoint_precision = 2 ** floating_point_precision

        self.w = None
        self.start_epoch = 0
        self.end_epoch = -1
        self.is_converged = False

    def asynchronous_compute_gradient(self, batch_ctx, encryptor, w, X, Y, weight):
        h = X.shape[0]
        # logger.info(f"h: {h}")
        Xw = torch.matmul(X, w.detach())
        half_d = 0.25 * Xw - 0.5 * Y
        if weight:
            half_d = half_d * weight
        batch_ctx.hosts.put("half_d", encryptor.encrypt_tensor(half_d, obfuscate=True))
        half_g = torch.matmul(X.T, half_d)

        Xw_h = batch_ctx.hosts.get("Xw_h")[0]
        if weight:
            Xw_h = Xw_h * weight

        if self.floating_point_precision:
            host_half_g = torch.matmul(torch.encode_as_int_f(X.T, self.floating_point_precision), Xw_h)
            host_half_g = 1 / self._fixpoint_precision * host_half_g
        else:
            host_half_g = torch.matmul(X.T, Xw_h)

        loss = np.log(2) - 1 + 0.125 / h * torch.matmul(Xw.T, Xw) - 2 / h * torch.matmul(half_d.T, Y)

        if self.optimizer.l1_penalty or self.optimizer.l2_penalty:
            loss_norm = self.optimizer.loss_norm(w)
            loss += loss_norm

        loss += torch.matmul((1 / h * Xw).T, Xw_h) - torch.matmul((2 / h * Y).T, Xw_h)

        for Xw2_h in batch_ctx.hosts.get("Xw2_h"):
            loss += 0.125 / h * Xw2_h
        h_loss_list = batch_ctx.hosts.get("h_loss")
        for h_loss in h_loss_list:
            if h_loss is not None:
                loss += h_loss

        batch_ctx.arbiter.put(loss=loss)
        # gradient
        g = 1 / h * (half_g + host_half_g)
        return g

    def centralized_compute_gradient(self, batch_ctx, w, X, Y, weight):
        h = X.shape[0]
        # logger.info(f"h: {h}")
        Xw = torch.matmul(X, w.detach())
        d = 0.25 * Xw - 0.5 * Y

        Xw_h_all = batch_ctx.hosts.get("Xw_h")

        for Xw_h in Xw_h_all:
            d += Xw_h

        if weight:
            # logger.info(f"weight: {weight.tolist()}")
            d = d * weight
        batch_ctx.hosts.put("d", d)

        # gradient
        if self.floating_point_precision:
            g = torch.matmul(torch.encode_as_int_f(X.T, self.floating_point_precision), d)
            g = 1 / (h * self._fixpoint_precision) * g
        else:
            g = 1 / h * torch.matmul(X.T, d)
        return g

    def fit_single_model(self, ctx: Context, encryptor, train_data, validate_data=None):
        """
        l(w) = 1/h * Σ(log(2) - 0.5 * y * xw + 0.125 * (wx)^2)
        ∇l(w) = 1/h * Σ(0.25 * xw - 0.5 * y)x = 1/h * Σdx
        where d = 0.25(xw - 2y)
        loss = log2 - (1/N)*0.5*∑ywx + (1/N)*0.125*[∑(Wg*Xg)^2 + ∑(Wh*Xh)^2 + 2 * ∑(Wg*Xg * Wh*Xh)]
        """
        coef_count = train_data.shape[1]
        if self.init_param.get("fit_intercept"):
            train_data["intercept"] = 1.0

        w = self.w
        if w is None:
            w = initialize_param(coef_count, **self.init_param)

            self.optimizer.init_optimizer(model_parameter_length=w.size()[0])
            self.lr_scheduler.init_scheduler(optimizer=self.optimizer.optimizer)

        train_data.label = train_data.label.apply_row(
            lambda x: [1.0] if abs(x[0] - 1) < 1e-8 else [-1.0], with_label=True
        )

        batch_loader = dataframe.DataLoader(
            train_data, ctx=ctx, batch_size=self.batch_size, mode="hetero", role="guest", sync_arbiter=True
        )
        # if self.end_epoch >= 0:
        #    self.start_epoch = self.end_epoch + 1

        is_centralized = len(ctx.hosts) > 1

        for i, iter_ctx in ctx.on_iterations.ctxs_range(self.epochs):
            self.optimizer.set_iters(i)
            logger.info(f"self.optimizer set epoch {i}")
            for batch_ctx, batch_data in iter_ctx.on_batches.ctxs_zip(batch_loader):
                X = batch_data.x
                Y = batch_data.label
                weight = batch_data.weight
                if is_centralized:
                    g = self.centralized_compute_gradient(batch_ctx, w, X, Y, weight)
                else:
                    g = self.asynchronous_compute_gradient(batch_ctx, encryptor, w, X, Y, weight)

                g = self.optimizer.add_regular_to_grad(g, w, self.init_param.get("fit_intercept"))
                batch_ctx.arbiter.put("g_enc", g)
                g = batch_ctx.arbiter.get("g")

                w = self.optimizer.update_weights(w, g, self.init_param.get("fit_intercept"), self.lr_scheduler.lr)
                # logger.info(f"w={w}")
                check_overflow(w)

            self.is_converged = iter_ctx.arbiter("converge_flag").get()
            if self.is_converged:
                self.end_epoch = i
                break
            if i < self.epochs - 1:
                logger.info(f"lr step at {i}th epoch")
                self.lr_scheduler.step()
        if not self.is_converged:
            self.end_epoch = self.epochs
        self.w = w
        logger.debug(f"Finish training at {self.end_epoch}th epoch.")

    def predict(self, ctx, test_data):
        if self.init_param.get("fit_intercept"):
            test_data["intercept"] = 1.0
        X = test_data.values.as_tensor()
        # logger.info(f"in predict, w: {self.w}")
        pred = torch.matmul(X, self.w.detach())
        for h_pred in ctx.hosts.get("h_pred"):
            pred += h_pred
        pred = torch.sigmoid(pred)
        return pred

    def get_model(self):
        param = serialize_param(self.w, self.init_param.get("fit_intercept"))
        return {
            # "w": w,
            # "intercept": intercept,
            "param": param,
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "end_epoch": self.end_epoch,
            "is_converged": self.is_converged,
            "fit_intercept": self.init_param.get("fit_intercept"),
        }

    def restore(self, model):
        self.w = deserialize_param(model["param"], model["fit_intercept"])
        self.optimizer = Optimizer()
        self.lr_scheduler = LRScheduler()
        self.optimizer.load_state_dict(model["optimizer"])
        self.lr_scheduler.load_state_dict(model["lr_scheduler"], self.optimizer.optimizer)
        self.end_epoch = model["end_epoch"]
        self.is_converged = model["is_converged"]
