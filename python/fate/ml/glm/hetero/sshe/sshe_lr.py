#
#  Copyright 2023 The FATE Authors. All Rights Reserved.
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

from fate.arch import Context, dataframe
from fate.arch.dataframe import DataFrame
from fate.arch.protocol.mpc.nn.sshe.lr_layer import (
    SSHELogisticRegressionLayer,
    SSHELogisticRegressionLossLayer,
    SSHEOptimizerSGD,
)
from fate.ml.abc.module import Module, HeteroModule
from fate.ml.utils import predict_tools
from fate.ml.utils._convergence import converge_func_factory
from fate.ml.utils._model_param import get_initialize_func
from fate.ml.utils._model_param import serialize_param, deserialize_param

logger = logging.getLogger(__name__)


class SSHELogisticRegression(Module):
    def __init__(
        self,
        epochs,
        batch_size,
        tol,
        early_stop,
        learning_rate,
        init_param,
        reveal_every_epoch=False,
        reveal_loss_freq=1,
        threshold=0.5,
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.tol = tol
        self.early_stop = early_stop
        self.learning_rate = learning_rate
        self.init_param = init_param
        self.threshold = threshold
        if reveal_every_epoch:
            raise ValueError(f"reveal_every_epoch is currenly not supported in SSHELogisticRegression")
        self.reveal_every_epoch = reveal_every_epoch
        self.reveal_loss_freq = reveal_loss_freq

        self.estimator = None
        self.ovr = False
        self.labels = None
        self.label_count = None

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.estimator.batch_size = batch_size

    def set_epochs(self, epochs):
        self.epochs = epochs
        self.estimator.epochs = epochs

    def fit(self, ctx: Context, train_data: DataFrame, validate_data=None):
        if len(ctx.hosts) > 1:
            raise ValueError(f"SSHE LR only support single-host case. Please check configuration.")
        if ctx.is_on_guest:
            train_data_binarized_label = train_data.label.get_dummies()
            label_count = train_data_binarized_label.shape[1]
            ctx.hosts.put("label_count", label_count)
            labels = [int(label_name.split("_")[1]) for label_name in train_data_binarized_label.columns]
            if self.labels is None:
                self.labels = sorted(labels)
        else:
            self.init_param["fit_intercept"] = False
            label_count = ctx.guest.get("label_count")
        self.label_count = label_count
        if label_count > 2 or self.ovr:
            logger.info(f"OVR data provided, will train OVR models.")
            self.ovr = True
            warm_start = True
            if self.estimator is None:
                self.estimator = {}
                warm_start = False
            for i, class_ctx in ctx.sub_ctx("class").ctxs_range(label_count):
                logger.info(f"start train for {i}th class")
                if not warm_start:
                    single_estimator = SSHELREstimator(
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        learning_rate=self.learning_rate,
                        init_param=self.init_param,
                        reveal_every_epoch=self.reveal_every_epoch,
                        reveal_loss_freq=self.reveal_loss_freq,
                        early_stop=self.early_stop,
                        tol=self.tol,
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
                if ctx.is_on_guest:
                    class_train_data.label = train_data_binarized_label[train_data_binarized_label.columns[i]]
                single_estimator.fit_single_model(class_ctx, class_train_data, class_validate_data)

                self.estimator[i] = single_estimator

        else:
            if self.estimator is None:
                single_estimator = SSHELREstimator(
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                    init_param=self.init_param,
                    reveal_every_epoch=self.reveal_every_epoch,
                    reveal_loss_freq=self.reveal_loss_freq,
                    early_stop=self.early_stop,
                    tol=self.tol,
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
            single_estimator.fit_single_model(ctx, train_data_fit, validate_data_fit)
            self.estimator = single_estimator

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
                "learning_rate": self.learning_rate,
                "init_param": self.init_param,
                "early_stop": self.early_stop,
                # "optimizer_param": self.optimizer_param,
                "labels": self.labels,
                "label_count": self.label_count,
                "ovr": self.ovr,
                "threshold": self.threshold,
                "reveal_every_epoch": self.reveal_every_epoch,
                "reveal_loss_freq": self.reveal_loss_freq,
                "tol": self.tol,
            },
        }

    @classmethod
    def from_model(cls, model):
        lr = SSHELogisticRegression(
            epochs=model["meta"]["epochs"],
            batch_size=model["meta"]["batch_size"],
            learning_rate=model["meta"]["learning_rate"],
            threshold=model["meta"]["threshold"],
            init_param=model["meta"]["init_param"],
            reveal_every_epoch=model["meta"]["reveal_every_epoch"],
            reveal_loss_freq=model["meta"]["reveal_loss_freq"],
            tol=model["meta"]["tol"],
            early_stop=model["meta"]["early_stop"],
        )
        lr.ovr = model["meta"]["ovr"]
        lr.labels = model["meta"]["labels"]
        lr.label_count = model["meta"]["label_count"]

        all_estimator = model["data"]["estimator"]
        lr.estimator = {}
        if lr.ovr:
            for label, d in all_estimator.items():
                estimator = SSHELREstimator(
                    epochs=model["meta"]["epochs"],
                    batch_size=model["meta"]["batch_size"],
                    init_param=model["meta"]["init_param"],
                    reveal_every_epoch=model["meta"]["reveal_every_epoch"],
                    reveal_loss_freq=model["meta"]["reveal_loss_freq"],
                    tol=model["meta"]["tol"],
                    early_stop=model["meta"]["early_stop"],
                )
                estimator.restore(d)
                lr.estimator[int(label)] = estimator
        else:
            estimator = SSHELREstimator(
                epochs=model["meta"]["epochs"],
                batch_size=model["meta"]["batch_size"],
                init_param=model["meta"]["init_param"],
                reveal_every_epoch=model["meta"]["reveal_every_epoch"],
                reveal_loss_freq=model["meta"]["reveal_loss_freq"],
                tol=model["meta"]["tol"],
                early_stop=model["meta"]["early_stop"],
            )
            estimator.restore(all_estimator)
            lr.estimator = estimator

        return lr

    def predict(self, ctx, test_data) -> DataFrame:
        if ctx.is_on_guest:
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
        else:
            if self.ovr:
                for i, class_ctx in ctx.sub_ctx("class").ctxs_range(self.label_count):
                    estimator = self.estimator[i]
                    estimator.predict(class_ctx, test_data)
            else:
                self.estimator.predict(ctx, test_data)


class SSHELREstimator(HeteroModule):
    def __init__(
        self,
        epochs=None,
        batch_size=None,
        optimizer=None,
        learning_rate=None,
        init_param=None,
        reveal_every_epoch=True,
        reveal_loss_freq=3,
        early_stop=None,
        tol=None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = learning_rate
        self.init_param = init_param
        self.reveal_every_epoch = reveal_every_epoch
        self.reveal_loss_freq = reveal_loss_freq
        self.early_stop = early_stop
        self.tol = tol

        self.w = None
        self.start_epoch = 0
        self.end_epoch = -1
        self.is_converged = False
        self.header = None
        self.converge_func = None
        if early_stop is not None:
            self.converge_func = converge_func_factory(self.early_stop, self.tol)

    def fit_single_model(self, ctx: Context, train_data: DataFrame, valid_data: DataFrame) -> None:
        self.header = train_data.schema.columns.to_list()
        rank_a, rank_b = ctx.hosts[0].rank, ctx.guest.rank
        if ctx.is_on_host:
            self.init_param["fit_intercept"] = False
        if self.w is None:
            initialize_func = get_initialize_func(**self.init_param)
        else:
            initialize_func = lambda x: self.w
        if self.init_param.get("fit_intercept"):
            train_data["intercept"] = 1.0
        train_data_n = train_data.shape[0]
        layer = SSHELogisticRegressionLayer(
            ctx,
            in_features_a=ctx.mpc.option_call(lambda: train_data.shape[1], dst=rank_a),
            in_features_b=ctx.mpc.option_call(lambda: train_data.shape[1], dst=rank_b),
            out_features=1,
            rank_a=rank_a,
            rank_b=rank_b,
            wa_init_fn=initialize_func,
            wb_init_fn=initialize_func,
        )
        loss_fn = SSHELogisticRegressionLossLayer(ctx, rank_a=rank_a, rank_b=rank_b)
        optimizer = SSHEOptimizerSGD(ctx, layer.parameters(), lr=self.lr)
        wa = layer.wa
        wb = layer.wb
        if ctx.is_on_guest:
            batch_loader = dataframe.DataLoader(
                train_data, ctx=ctx, batch_size=self.batch_size, mode="hetero", role="guest", sync_arbiter=False
            )
        else:
            batch_loader = dataframe.DataLoader(
                train_data, ctx=ctx, batch_size=self.batch_size, mode="hetero", role="host"
            )
        # if self.reveal_every_epoch:
        if self.early_stop == "weight_diff":
            wa_p = wa.get_plain_text(dst=rank_a)
            wb_p = wb.get_plain_text(dst=rank_b)
            if ctx.is_on_guest:
                self.converge_func.set_pre_weight(wb_p)
            else:
                self.converge_func.set_pre_weight(wa_p)
        for i, epoch_ctx in ctx.on_iterations.ctxs_range(self.epochs):
            epoch_loss = None
            logger.info(f"enter {i}th epoch")
            for batch_ctx, batch_data in epoch_ctx.on_batches.ctxs_zip(batch_loader):
                h = batch_data.x
                y = batch_ctx.mpc.cond_call(lambda: batch_data.label, lambda: None, dst=rank_b)
                # if self.reveal_every_epoch:
                #    z = batch_ctx.mpc.cond_call(lambda: torch.matmul(h, wa_p.detach()),
                #                                lambda: torch.matmul(h, wb_p.detach()), dst=rank_a)
                # else:
                z = layer(h)
                loss = loss_fn(z, y)
                if i % self.reveal_loss_freq == 0:
                    if epoch_loss is None:
                        epoch_loss = loss.get(dst=rank_b)
                        if epoch_loss:
                            epoch_loss = epoch_loss * h.shape[0]
                    else:
                        batch_loss = loss.get(dst=rank_b)
                        if batch_loss:
                            epoch_loss += batch_loss * h.shape[0]
                loss.backward()
                optimizer.step()
            if epoch_loss is not None and ctx.is_on_guest:
                epoch_loss = epoch_loss / train_data_n
                epoch_ctx.metrics.log_loss("lr_loss", epoch_loss.tolist())
            # if self.reveal_every_epoch:
            #    wa_p = wa.get_plain_text(dst=rank_a)
            #    wb_p = wb.get_plain_text(dst=rank_b)
            if ctx.is_on_guest:
                if self.early_stop == "weight_diff":
                    """if self.reveal_every_epoch:
                        wb_p_delta = self.converge_func.compute_weight_diff(wb_p - self.converge_func.pre_weight)
                        w_diff = wb_p_delta + epoch_ctx.hosts.get("wa_p_delta")[0]
                        self.converge_func.set_pre_weight(wb_p)
                        if w_diff < self.tol:
                            self.is_converged = True
                    else:
                        raise ValueError(f"early stop {self.early_stop} is not supported when "
                                         f"reveal_every_epoch is False")"""
                    wa_p = wa.get_plain_text(dst=rank_a)
                    wb_p = wb.get_plain_text(dst=rank_b)
                    wb_p_delta = self.converge_func.compute_weight_diff(wb_p - self.converge_func.pre_weight)
                    w_diff = wb_p_delta + epoch_ctx.hosts.get("wa_p_delta")[0]
                    self.converge_func.set_pre_weight(wb_p)
                    if w_diff < self.tol:
                        self.is_converged = True
                else:
                    if i % self.reveal_loss_freq == 0:
                        self.is_converged = self.converge_func.is_converge(epoch_loss)
                epoch_ctx.hosts.put("converge_flag", self.is_converged)
            else:
                if self.early_stop == "weight_diff":
                    # if self.reveal_every_epoch:
                    wa_p = wa.get_plain_text(dst=rank_a)
                    wb_p = wb.get_plain_text(dst=rank_b)
                    wa_p_delta = self.converge_func.compute_weight_diff(wa_p - self.converge_func.pre_weight)
                    epoch_ctx.guest.put("wa_p_delta", wa_p_delta)
                    self.converge_func.set_pre_weight(wa_p)
                self.is_converged = epoch_ctx.guest.get("converge_flag")
            if self.is_converged:
                self.end_epoch = i
                break
        if not self.is_converged:
            self.end_epoch = self.epochs
        wa_p = wa.get_plain_text(dst=rank_a)
        wb_p = wb.get_plain_text(dst=rank_b)
        if ctx.is_on_host:
            self.w = wa_p
        else:
            self.w = wb_p

    def predict(self, ctx, test_data):
        if ctx.is_on_guest:
            if self.init_param.get("fit_intercept"):
                test_data["intercept"] = 1.0
            X = test_data.values.as_tensor()
            # logger.info(f"in predict, w: {self.w}")
            pred = torch.matmul(X, self.w.detach())
            h_pred = ctx.hosts.get("h_pred")[0]
            pred += h_pred
            pred = torch.sigmoid(pred)

            return pred
        else:
            X = test_data.values.as_tensor()
            output = torch.matmul(X, self.w)
            ctx.guest.put("h_pred", output)

    def get_model(self):
        param = serialize_param(self.w, self.init_param.get("fit_intercept"))
        return {
            "param": param,
            # "optimizer": self.optimizer.state_dict(),
            "end_epoch": self.end_epoch,
            "is_converged": self.is_converged,
            "fit_intercept": self.init_param.get("fit_intercept"),
            "header": self.header,
            "lr": self.lr,
        }

    def restore(self, model):
        self.w = deserialize_param(model["param"], model["fit_intercept"])
        self.end_epoch = model["end_epoch"]
        self.is_converged = model["is_converged"]
        self.header = model["header"]
        self.init_param["fit_intercept"] = model["fit_intercept"]
        self.lr = model["lr"]
        # self.optimizer.load_state_dict(model["optimizer"])
