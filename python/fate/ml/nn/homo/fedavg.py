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

from fate.ml.nn.trainer.trainer_base import HomoTrainerClient, HomoTrainerServer
from fate.ml.nn.trainer.trainer_base import FedArguments, TrainingArguments
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
from fate.arch import Context
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import _LRScheduler
from transformers.trainer_callback import TrainerCallback
from torch.nn import Module
from torch import nn
from torch.utils.data import DataLoader
from transformers import TrainerState, TrainerControl, PreTrainedTokenizer
from fate.ml.aggregator import AggregatorClientWrapper, AggregatorServerWrapper
import logging


logger = logging.getLogger(__name__)


@dataclass
class FedAVGArguments(FedArguments):
    pass


class FedAVGClient(HomoTrainerClient):
    def __init__(
        self,
        ctx: Context,
        model: Module,
        training_args: TrainingArguments,
        fed_args: FedArguments,
        train_set: Dataset,
        val_set: Dataset = None,
        loss_fn: Module = None,
        optimizer: Optimizer = None,
        scheduler: _LRScheduler = None,
        callbacks: List[TrainerCallback] = [],
        data_collator: Callable = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        use_hf_default_behavior: bool = False,
        compute_metrics: Callable = None,
        local_mode: bool = False,
    ):
        super().__init__(
            ctx,
            model,
            training_args,
            fed_args,
            train_set,
            val_set,
            loss_fn,
            optimizer,
            data_collator,
            scheduler,
            tokenizer,
            callbacks,
            use_hf_default_behavior,
            compute_metrics=compute_metrics,
            local_mode=local_mode,
        )

    def init_aggregator(self, ctx: Context, fed_args: FedArguments):
        aggregate_type = "weighted_mean"
        aggregator_name = "fedavg"
        aggregator = fed_args.aggregator
        return AggregatorClientWrapper(
            ctx, aggregate_type, aggregator_name, aggregator, sample_num=len(self.train_dataset), args=self._args
        )

    def on_federation(
        self,
        ctx: Context,
        aggregator: AggregatorClientWrapper,
        fed_args: FedArguments,
        args: TrainingArguments,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        dataloader: Optional[Tuple[DataLoader]] = None,
        control: Optional[TrainerControl] = None,
        state: Optional[TrainerState] = None,
        **kwargs,
    ):
        aggregator.model_aggregation(ctx, model)


class FedAVGServer(HomoTrainerServer):
    def __init__(self, ctx: Context, local_mode: bool = False) -> None:
        super().__init__(ctx, local_mode)

    def init_aggregator(self, ctx):
        return AggregatorServerWrapper(ctx)

    def on_federation(self, ctx: Context, aggregator: AggregatorServerWrapper, agg_iter_idx: int):
        aggregator.model_aggregation(ctx)


class FedAVG(object):
    client = FedAVGClient
    server = FedAVGServer
