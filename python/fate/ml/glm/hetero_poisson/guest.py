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

import json
import logging

import torch

from fate.arch import dataframe
from fate.interface import Context
from fate.ml.abc.module import HeteroModule
from fate.ml.utils._model_param import initialize_param
from fate.ml.utils._optimizer import Optimizer, LRScheduler

logger = logging.getLogger(__name__)


class HeteroPoissonModuleGuest(HeteroModule):
    def __init__(
            self,
            max_iter=None,
            batch_size=None,
            optimizer_param=None,
            learning_rate_param=None,
            init_param=None,
            exposure_col_name=None
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
        self.exposure_col_name = exposure_col_name

        self.estimator = None

    def fit(self, ctx: Context, train_data, validate_data=None) -> None:
        with_weight = train_data.weight is not None

        estimator = HeteroPoissonEstimatorGuest(max_iter=self.max_iter,
                                                batch_size=self.batch_size,
                                                optimizer=self.optimizer,
                                                learning_rate_scheduler=self.lr_scheduler,
                                                init_param=self.init_param,
                                                exposure_col_name=self.exposure_col_name)
        estimator.fit_model(ctx, train_data, validate_data, with_weight=with_weight)
        self.estimator = estimator

    def predict(self, ctx, test_data):
        df = test_data.create_dataframe(with_label=True, with_weight=False)
        prob = self.estimator.predict(test_data)
        pred_res = test_data.create_dataframe(with_label=False, with_weight=False)
        pred_res["predict_result"] = prob
        df[["predict_result", "predict_score", "predict_detail"]] = pred_res.apply_row(lambda v: [
            v[0],
            v[0],
            json.dumps({v[0]})],
                                                                                       enable_type_align_checking=False)
        return df

    def get_model(self):
        return {
            "estimator": self.estimator.get_model(),
        }

    @classmethod
    def from_model(cls, model) -> "HeteroPoissonModuleGuest":
        poisson = HeteroPoissonModuleGuest(**model["metadata"])
        estimator = HeteroPoissonEstimatorGuest()
        estimator.restore(model["estimator"])
        poisson.estimator = estimator

        return poisson


class HeteroPoissonEstimatorGuest(HeteroModule):
    def __init__(
            self,
            max_iter=None,
            batch_size=None,
            optimizer=None,
            learning_rate_scheduler=None,
            init_param=None,
            exposure_col_name=None
    ):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr_scheduler = learning_rate_scheduler
        self.init_param = init_param
        self.exposure_col_name = exposure_col_name

        self.w = None
        self.start_iter = 0
        self.end_iter = -1
        self.is_converged = False

    def fit_model(self, ctx, train_data, validate_data=None, with_weight=False):
        coef_count = train_data.shape[1]
        if self.init_param.fit_intercept:
            train_data["intercept"] = 1
            coef_count += 1
        if self.exposure_col_name is not None:
            header = train_data.schema.columns.to_list()
            select_cols = [col for col in header if col != self.exposure_col_name]
            new_header = select_cols + [self.exposure_col_name]
            train_data = train_data[new_header]
            coef_count -= 1
        else:
            train_data[f"exposure_col"] = 1

        w = self.w
        if self.w is None:
            w = initialize_param(coef_count, **self.init_param)
            self.optimizer.init_optimizer(model_parameter_length=w.size()[0])
        batch_loader = dataframe.DataLoader(
            train_data, ctx=ctx, batch_size=self.batch_size, mode="hetero", role="guest", sync_arbiter=True,
            return_weight=False
        )
        if self.end_iter >= 0:
            self.start_iter = self.end_iter + 1
        for i, iter_ctx in ctx.range(self.start_iter, self.max_iter):
            logger.info(f"start iter {i}")
            j = 0
            self.optimizer.set_iters(i)
            for batch_ctx, (X, Y) in iter_ctx.iter(batch_loader):

                # extract exposure col
                X = X[:, :-1]
                exposure_col = X[:, -1]

                h = X.shape[0]
                Xw = torch.matmul(X, w)
                mu = torch.exp(Xw) * exposure_col
                mu_h = batch_ctx.hosts.get("mu_h")[0]
                mu_total = mu * mu_h
                d = mu_total - Y

                Xw_h = batch_ctx.hosts.get("Xw_h")[0]

                loss = 1 / h * (mu_total.sum() - torch.matmul(Y.T, (Xw + Xw_h + torch.log(exposure_col))))

                if self.optimizer.l1_penalty or self.optimizer.l2_penalty:
                    loss_norm = self.optimizer.loss_norm(w)
                    loss += loss_norm

                batch_ctx.hosts.put(d=d)
                h_loss_list = batch_ctx.hosts.get("h_loss")
                for h_loss in h_loss_list:
                    if h_loss is not None:
                        loss += h_loss
                batch_ctx.arbiter.put(loss=loss)

                # gradient
                g = self.optimizer.add_regular_to_grad(X.T @ d, w, self.init_param.fit_intercept)
                batch_ctx.arbiter.put("g_enc", X.T @ g)
                g = batch_ctx.arbiter.get("g")

                w = self.optimizer.update_weights(w, g, self.init_param.fit_intercept, self.lr_scheduler.lr)
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
        if self.exposure_col_name is not None:
            header = test_data.schema.columns.to_list()
            select_cols = [col for col in header if col != self.exposure_col_name]
            new_header = select_cols + [self.exposure_col_name]
            test_data = test_data[new_header]
        else:
            test_data[f"exposure_col"] = 1

        X = test_data.values.as_tensor()
        X = X[:, :-1]
        exposure_col = X[:, -1]
        Xw = torch.matmul(X, self.w)
        h_pred = ctx.hosts.get("h_pred")[0]
        pred = torch.exp(Xw + h_pred) * exposure_col

        return pred

    def get_model(self):
        return {
            "w": self.w.tolist(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "end_iter": self.end_iter,
            "converged": self.is_converged,
            "exposure_col_name": self.exposure_col_name
        }

    def restore(self, model):

        self.w = torch.tensor(model["w"])
        self.optimizer.load_state_dict(model["optimizer"])
        self.lr_scheduler.load_state_dict(model["lr_scheduler"])
        self.end_iter = model["end_iter"]
        self.is_converged = model["is_converged"]
        self.exposure_col_name = model["exposure_col_name"]
