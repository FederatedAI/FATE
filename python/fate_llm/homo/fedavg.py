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
from fate.ml.nn.homo.fedavg import FedAVGServer, FedAVGArguments, FedArguments
from fate.arch import Context
from fate_llm.trainer.seq2seq_trainer import HomoSeq2SeqTrainerClient, Seq2SeqTrainingArguments
from fate.ml.aggregator import AggregatorClientWrapper
import logging
from typing import List, Optional, Tuple, Callable, Dict
from fate.arch import Context
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import _LRScheduler
from transformers.trainer_callback import TrainerCallback
from torch import nn
from torch.utils.data import DataLoader
from transformers import TrainerState, TrainerControl, PreTrainedTokenizer, EvalPrediction


logger = logging.getLogger(__name__)


Seq2SeqFedAVGServer = FedAVGServer


class Seq2SeqFedAVGClient(HomoSeq2SeqTrainerClient):

    def __init__(
        self,
        ctx: Context,
        model: nn.Module,
        training_args: Seq2SeqTrainingArguments,
        fed_args: FedArguments,
        train_set: Dataset,
        val_set: Dataset = None,
        optimizer: torch.optim.Optimizer = None,
        data_collator: Callable = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        callbacks: Optional[List[TrainerCallback]] = [],
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        local_mode: bool = False
    ):
        # in case you forget to set evaluation_strategy
        if val_set is not None and training_args.evaluation_strategy == "no":
            training_args.evaluation_strategy = "epoch"

        HomoSeq2SeqTrainerClient.__init__(
            self,
            ctx,
            model,
            training_args,
            fed_args,
            train_set,
            val_set,
            optimizer,
            data_collator,
            scheduler,
            tokenizer,
            callbacks,
            compute_metrics,
            local_mode
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
        args: Seq2SeqTrainingArguments,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        dataloader: Optional[Tuple[DataLoader]] = None,
        control: Optional[TrainerControl] = None,
        state: Optional[TrainerState] = None,
        **kwargs,
    ):
        aggregator.model_aggregation(ctx, model)

