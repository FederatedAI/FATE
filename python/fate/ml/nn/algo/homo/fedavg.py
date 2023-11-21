from fate.ml.nn.trainer.trainer_base import FedTrainerClient, FedTrainerServer
from fate.ml.nn.trainer.trainer_base import FedArguments, TrainingArguments
from dataclasses import field
from dataclasses import dataclass, field
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable, Union
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


class FedAVGCLient(FedTrainerClient):
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

    def init_aggregator(self, ctx: Context, training_args: TrainingArguments, fed_args: FedArguments):
        aggregate_type = "weighted_mean"
        aggregator_name = "fedavg"
        aggregator = fed_args.aggregator

        return AggregatorClientWrapper(
            ctx,
            aggregate_type,
            aggregator_name,
            aggregator,
            sample_num=len(self.train_dataset),
            args=self._args
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


class FedAVGServer(FedTrainerServer):
    def __init__(self, ctx: Context, local_mode: bool = False) -> None:
        super().__init__(ctx, local_mode)

    def init_aggregator(self, ctx):
        return AggregatorServerWrapper(ctx)

    def on_federation(self, ctx: Context, aggregator: AggregatorServerWrapper):
        aggregator.model_aggregation(ctx)


class FedAVG(object):
    client = FedAVGCLient
    server = FedAVGServer
