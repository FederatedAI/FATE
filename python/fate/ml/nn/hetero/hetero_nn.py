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

import torch
from dataclasses import dataclass, fields
from enum import Enum
from torch import nn
from typing import Any, Dict, List, Union, Callable, Literal
from fate.arch import Context
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from transformers import EvalPrediction
from transformers.trainer_callback import TrainerCallback
from typing import Optional
from fate.ml.nn.model_zoo.hetero_nn_model import HeteroNNModelGuest, HeteroNNModelHost
from fate.ml.nn.trainer.trainer_base import HeteroTrainerBase, TrainingArguments
from fate.ml.nn.model_zoo.hetero_nn_model import StdAggLayerArgument, FedPassArgument, SSHEArgument
from fate.ml.nn.model_zoo.hetero_nn_model import TopModelStrategyArguments


class HeteroNNTrainerGuest(HeteroTrainerBase):
    def __init__(
        self,
        ctx: Context,
        model: HeteroNNModelGuest,
        training_args: TrainingArguments,
        train_set: Dataset,
        val_set: Dataset = None,
        loss_fn: nn.Module = None,
        optimizer=None,
        data_collator: Callable = None,
        scheduler=None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        callbacks: Optional[List[TrainerCallback]] = [],
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    ):
        assert isinstance(model, HeteroNNModelGuest), (
            "Model should be a HeteroNNModelGuest instance, " "but got {}."
        ).format(type(model))

        if model.need_mpc_init():
            ctx.mpc.init()

        model.setup(ctx=ctx)

        super().__init__(
            ctx=ctx,
            model=model,
            training_args=training_args,
            train_set=train_set,
            val_set=val_set,
            loss_fn=loss_fn,
            optimizer=optimizer,
            data_collator=data_collator,
            scheduler=scheduler,
            tokenizer=tokenizer,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )

    def compute_loss(self, model, inputs, **kwargs):
        # (features, labels), this format is used in FATE-1.x
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if len(inputs) == 2:  # data & label
                feats, labels = inputs
                output = model(feats)
                loss = self.loss_func(output, labels)
                return loss
            if len(inputs) == 1:  # label only
                labels = inputs[0]
                output = model()
                loss = self.loss_func(output, labels)
                return loss
        else:
            # unknown format, go to super class function
            return super().compute_loss(model, inputs, **kwargs)

    def training_step(
        self, model: Union[HeteroNNModelGuest, HeteroNNModelHost], inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        # override the training_step method in Trainer
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        model.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        # (features, labels), this format is used in FATE-1.x
        # now the model is in eval status
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                if len(inputs) == 2:  # data & label
                    feats, labels = inputs
                    output = model(feats)
                if len(inputs) == 1:  # label only
                    labels = inputs[0]
                    output = model()
            return None, output, labels
        else:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)


class HeteroNNTrainerHost(HeteroTrainerBase):
    def __init__(
        self,
        ctx: Context,
        model: HeteroNNModelHost,
        training_args: TrainingArguments,
        train_set: Dataset,
        val_set: Dataset = None,
        optimizer=None,
        data_collator: Callable = None,
        scheduler=None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        callbacks: Optional[List[TrainerCallback]] = [],
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
    ):
        assert isinstance(model, HeteroNNModelHost), (
            "Model should be a HeteroNNModelHost instance, " "but got {}."
        ).format(type(model))

        if model.need_mpc_init():
            ctx.mpc.init()

        model.setup(ctx=ctx)
        super().__init__(
            ctx=ctx,
            model=model,
            training_args=training_args,
            train_set=train_set,
            val_set=val_set,
            loss_fn=None,
            optimizer=optimizer,
            data_collator=data_collator,
            scheduler=scheduler,
            tokenizer=tokenizer,
            callbacks=callbacks,
            compute_metrics=compute_metrics,
        )

    def compute_loss(self, model, inputs, **kwargs):
        # host side not computing loss
        if isinstance(inputs, torch.Tensor):
            feats = inputs
        elif isinstance(inputs, tuple) or isinstance(inputs, list):
            feats = inputs[0]
        else:
            return super().compute_loss(model, inputs, **kwargs)
        model(feats)
        return 0

    def training_step(
        self, model: Union[HeteroNNModelGuest, HeteroNNModelHost], inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        # override the training_step method in Trainer
        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            self.compute_loss(model, inputs)
        model.backward()
        # host has no label, will never have loss
        return torch.tensor(0)

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        # (features, labels), this format is used in FATE-1.x
        # now the model is in eval status
        inputs = self._prepare_inputs(inputs)
        if isinstance(inputs, torch.Tensor):
            feats = inputs
        elif isinstance(inputs, tuple) or isinstance(inputs, list):
            feats = inputs[0]
        else:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        with torch.no_grad():
            model(feats)
        return None, None, None
