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
from fate.ml.utils._optimizer import LRScheduler, Optimizer, separate

logger = logging.getLogger(__name__)


class CoordinatedLinRModuleArbiter(HeteroModule):
    def __init__(self, epochs, early_stop, tol, batch_size, optimizer_param, learning_rate_param):
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.tol = tol
        self.learning_rate_param = learning_rate_param
        self.optimizer_param = optimizer_param

        self.estimator = None

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.estimator.batch_size = batch_size

    def set_epochs(self, epochs):
        self.epochs = epochs
        self.estimator.epochs = epochs

    def fit(self, ctx: Context) -> None:
        kit = ctx.cipher.phe.setup()
        encryptor = kit.get_tensor_encryptor()
        decryptor = kit.get_tensor_decryptor()
        ctx.hosts("encryptor").put(encryptor)
        ctx.guest("encryptor").put(encryptor)
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
            single_estimator = HeteroLinREstimatorArbiter(
                epochs=self.epochs,
                early_stop=self.early_stop,
                tol=self.tol,
                batch_size=self.batch_size,
                optimizer=optimizer,
                learning_rate_scheduler=lr_scheduler,
            )
            self.estimator = single_estimator
        self.estimator.fit_model(ctx, decryptor)

    def get_model(self):
        return {
            "data": {"estimator": self.estimator.get_model()},
            "meta": {
                "epochs": self.epochs,
                "early_stop": self.early_stop,
                "tol": self.tol,
                "batch_size": self.batch_size,
                "learning_rate_param": self.learning_rate_param,
                "optimizer_param": self.optimizer_param,
            },
        }

    @classmethod
    def from_model(cls, model):
        linr = CoordinatedLinRModuleArbiter(
            model["meta"]["epochs"],
            model["meta"]["early_stop"],
            model["meta"]["tol"],
            model["meta"]["batch_size"],
            model["meta"]["optimizer_param"],
            model["meta"]["learning_rate_param"],
        )
        estimator = HeteroLinREstimatorArbiter()
        estimator.restore(model["data"]["estimator"])
        linr.estimator = estimator
        return linr


class HeteroLinREstimatorArbiter(HeteroModule):
    def __init__(
        self, epochs=None, early_stop=None, tol=None, batch_size=None, optimizer=None, learning_rate_scheduler=None
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.tol = tol
        self.optimizer = optimizer
        self.lr_scheduler = learning_rate_scheduler

        if early_stop is not None:
            self.converge_func = converge_func_factory(early_stop, tol)
        self.start_epoch = 0
        self.end_epoch = -1
        self.is_converged = False

    def fit_model(self, ctx, decryptor):
        batch_loader = DataLoader(
            dataset=None, ctx=ctx, batch_size=self.batch_size, mode="hetero", role="arbiter", sync_arbiter=True
        )
        logger.info(f"batch_num={batch_loader.batch_num}")
        if self.optimizer.optimizer is None:
            optimizer_ready = False
        else:
            optimizer_ready = True
            # self.start_epoch = self.end_epoch + 1

        for i, iter_ctx in ctx.on_iterations.ctxs_range(self.epochs):
            iter_loss = None
            iter_g = None
            self.optimizer.set_iters(i)
            logger.info(f"self.optimizer set epoch {i}")
            for batch_ctx, _ in iter_ctx.on_batches.ctxs_zip(batch_loader):
                g_guest_enc = batch_ctx.guest.get("g_enc")
                g_guest = decryptor.decrypt_tensor(g_guest_enc)
                size_list = [g_guest.size()[0]]
                g_total = g_guest.squeeze()
                # get torch tensor

                host_g = batch_ctx.hosts.get("g_enc")
                for i, g_host_enc in enumerate(host_g):
                    g = decryptor.decrypt_tensor(g_host_enc)
                    size_list.append(g.size()[0])
                    g_total = torch.hstack((g_total, g.squeeze()))
                if not optimizer_ready:
                    self.optimizer.init_optimizer(model_parameter_length=sum(size_list), dtype=g_total.dtype)
                    self.lr_scheduler.init_scheduler(self.optimizer.optimizer)
                    optimizer_ready = True
                self.optimizer.step(g_total.unsqueeze(1))
                delta_g = self.optimizer.get_delta_gradients()
                delta_g_list = separate(delta_g, size_list)

                delta_g_list_squeezed = []
                batch_ctx.guest.put("g", delta_g_list[0])
                delta_g_list_squeezed.append(delta_g_list[0].squeeze())
                for i, g_host in enumerate(delta_g_list[1:]):
                    batch_ctx.hosts[i].put("g", g_host)
                    delta_g_list_squeezed.append(g_host.squeeze())
                if iter_g is None:
                    iter_g = torch.hstack(delta_g_list_squeezed)
                else:
                    iter_g += torch.hstack(delta_g_list_squeezed)

                if len(host_g) == 1:
                    loss = decryptor.decrypt_tensor(batch_ctx.guest.get("loss"))
                    iter_loss = 0 if iter_loss is None else iter_loss
                    iter_loss += loss
                else:
                    logger.info("Multiple hosts exist, do not compute loss.")

            if iter_loss is not None:
                iter_ctx.metrics.log_loss("linr_loss", iter_loss.tolist()[0])
            if self.early_stop == "weight_diff":
                self.is_converged = self.converge_func.is_converge(iter_g)
            else:
                if iter_loss is None:
                    raise ValueError(
                        "Multiple host situation, loss early stop function is not available."
                        "You should use 'weight_diff' instead"
                    )
                self.is_converged = self.converge_func.is_converge(iter_loss)

            iter_ctx.hosts.put("converge_flag", self.is_converged)
            iter_ctx.guest.put("converge_flag", self.is_converged)

            if self.is_converged:
                self.end_epoch = i
                break
            if i < self.epochs - 1:
                self.lr_scheduler.step()
        if not self.is_converged:
            self.end_epoch = self.epochs
        logger.debug(f"Finish training at {self.end_epoch}th epoch.")

    def get_model(self):
        return {
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "end_epoch": self.end_epoch,
            "is_converged": self.is_converged,
            "tol": self.tol,
            "early_stop": self.early_stop,
        }

    def restore(self, model):
        self.optimizer = Optimizer()
        self.lr_scheduler = LRScheduler()
        self.optimizer.load_state_dict(model["optimizer"])
        self.lr_scheduler.load_state_dict(model["lr_scheduler"], self.optimizer.optimizer)
        self.end_epoch = model["end_epoch"]
        self.is_converged = model["is_converged"]
        self.tol = model["tol"]
        self.early_stop = model["early_stop"]
        self.converge_func = converge_func_factory(self.early_stop, self.tol)
        # self.start_epoch = model["end_epoch"] + 1
