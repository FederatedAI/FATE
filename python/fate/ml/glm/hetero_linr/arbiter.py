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
from fate.ml.utils._convergence import converge_func_factory
from fate.ml.utils._optimizer import separate, Optimizer, LRScheduler

logger = logging.getLogger(__name__)


class HeteroLinRModuleArbiter(HeteroModule):
    def __init__(
            self,
            max_iter,
            early_stop,
            tol,
            batch_size,
            optimizer_param,
            learning_rate_param

    ):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.tol = tol
        self.optimizer = Optimizer(optimizer_param["method"],
                                   optimizer_param["penalty"],
                                   optimizer_param["alpha"],
                                   optimizer_param["optimizer_params"])
        self.lr_scheduler = LRScheduler(learning_rate_param["method"],
                                        learning_rate_param["scheduler_params"])
        """self.optimizer = Optimizer(optimizer_param.method,
                                   optimizer_param.penalty,
                                   optimizer_param.alpha,
                                   optimizer_param.optimizer_params)
        self.lr_scheduler = LRScheduler(learning_rate_param.method,
                                        learning_rate_param.scheduler_params)"""

        self.estimator = None

    def fit(self, ctx: Context) -> None:
        encryptor, decryptor = ctx.cipher.phe.keygen(options=dict(key_length=2048))
        ctx.hosts("encryptor").put(encryptor)
        single_estimator = HeteroLinrEstimatorArbiter(max_iter=self.max_iter,
                                                      early_stop=self.early_stop,
                                                      tol=self.tol,
                                                      batch_size=self.batch_size,
                                                      optimizer=self.optimizer,
                                                      learning_rate_scheduler=self.lr_scheduler)
        single_estimator.fit_model(ctx, decryptor)
        self.estimator = single_estimator

    def to_model(self):
        return {
            "estimator": self.estimator.get_model(),
        }

    def from_model(cls, model):
        linr = HeteroLinRModuleArbiter(**model["metadata"])
        estimator = HeteroLinrEstimatorArbiter()
        estimator.restore(model["estimator"])
        linr.estimator = estimator
        return linr


class HeteroLinrEstimatorArbiter(HeteroModule):
    def __init__(
            self,
            max_iter=None,
            early_stop=None,
            tol=None,
            batch_size=None,
            optimizer=None,
            learning_rate_scheduler=None

    ):
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.tol = tol
        self.optimizer = optimizer
        self.lr_scheduler = learning_rate_scheduler

        self.converge_func = converge_func_factory(early_stop, tol)
        self.start_iter = 0
        self.end_iter = -1
        self.is_converged = False

    def fit_model(self, ctx, decryptor):
        batch_loader = DataLoader(
            dataset=None, ctx=ctx, batch_size=self.batch_size, mode="hetero", role="arbiter", sync_arbiter=True
        )
        logger.info(f"batch_num={batch_loader.batch_num}")
        if self.optimizer is None:
            optimizer_ready = False
        else:
            optimizer_ready = True
            self.start_iter = self.end_iter + 1

        # for i, iter_ctx in ctx.range(self.start_iter, self.max_iter):
        # temp code start
        for i, iter_ctx in ctx.ctxs_range(self.max_iter):
            # temp code end
            logger.info(f"start iter {i}")
            iter_loss = None
            iter_g = None
            self.optimizer.set_iters(i)
            for batch_ctx, _ in iter_ctx.ctxs_zip(batch_loader):
                g_guest_enc = batch_ctx.guest.get("g_enc")
                g_guest = decryptor.decrypt(g_guest_enc)
                size_list = [g_guest.size()[0]]
                g_total = g_guest  # get torch tensor

                host_g = batch_ctx.hosts.get("g_enc")
                for i, g_host_enc in enumerate(host_g):
                    g = decryptor.decrypt(g_host_enc)
                    size_list.append(g.size()[0])
                    g_total = torch.hstack((g_total, g))
                if not optimizer_ready:
                    self.optimizer.init_optimizer(size_list)
                    self.lr_scheduler.init_scheduler(self.optimizer)
                    optimizer_ready = True
                self.optimizer.step(g_total)
                delta_g = self.optimizer.get_delta_gradients()
                delta_g_list = separate(delta_g, size_list)
                for i, g_host in enumerate(delta_g_list[1:]):
                    batch_ctx.hosts[i].put("g", g_host)
                batch_ctx.guest.put("g", delta_g_list[0])
                if iter_g is None:
                    iter_g = torch.hstack(delta_g_list)
                else:
                    iter_g += torch.hstack(delta_g_list)

                if len(host_g) == 1:
                    loss = decryptor.decrypt(batch_ctx.guest.get("loss"))
                    iter_loss = 0 if iter_loss is None else iter_loss
                    iter_loss += loss
                else:
                    logger.info("Multiple hosts exist, do not compute loss.")

            if iter_loss is not None:
                iter_ctx.metrics.log_loss("linr_loss", iter_loss.tolist(), step=i)
            if self.early_stop == 'weight_diff':
                self.is_converged = self.converge_func.is_converge(iter_g)
            else:
                if iter_loss is None:
                    raise ValueError("Multiple host situation, loss early stop function is not available."
                                     "You should use 'weight_diff' instead")
                self.is_converged = self.converge_func.is_converge(iter_loss)

            iter_ctx.hosts.put("converge_flag", self.is_converged)
            iter_ctx.guest.put("converge_flag", self.is_converged)

            if self.is_converged:
                self.end_iter = i
                break
            self.lr_scheduler.step()
        if not self.is_converged:
            self.end_iter = self.max_iter
        logger.debug(f"Finish training at {self.end_iter}th iteration.")

    def to_model(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "end_iter": self.end_iter,
            "converged": self.is_converged
        }

    def restore(self, model):
        self.optimizer.load_state_dict(model["optimizer"])
        self.lr_scheduler.load_state_dict(model["lr_scheduler"])
        self.end_iter = model["end_iter"]
        self.is_converged = model["is_converged"]
        # self.start_iter = model["end_iter"] + 1
