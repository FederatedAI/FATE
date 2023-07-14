from transformers.training_args import TrainingArguments
from fate.ml.aggregator.base import Aggregator
from fate.ml.nn.trainer.trainer_base import FedTrainerClient, FedTrainerServer, TrainingArguments
from fate.ml.nn.trainer.trainer_base import FedArguments, time_decorator, TrainingArguments
from dataclasses import field
from dataclasses import dataclass, field
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
from fate.ml.aggregator.plaintext_aggregator import PlainTextAggregatorClient, PlainTextAggregatorServer
from transformers import TrainerState, TrainerControl, PreTrainedTokenizer


@dataclass
class FedAVGArguments(FedArguments):

    """
    The arguemnt for FedAVG algorithm, used in FedAVGClient and FedAVGServer.

    Attributes:
        weighted_aggregate: bool
            Whether to use weighted aggregation or not.
        secure_aggregate: bool
            Whether to use secure aggregation or not.
    """

    weighted_aggregate: bool = field(default=True)
    secure_aggregate: bool = field(default=False)


class FedAVGCLient(FedTrainerClient):

    def __init__(self,
                 ctx: Context,
                 model: Module,
                 training_args: TrainingArguments, fed_args: FedArguments,
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
                 local_mode: bool = False
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
            local_mode=local_mode)

    def init_aggregator(self):
        sample_num = len(self.train_dataset)
        aggregator = PlainTextAggregatorClient(
            self.ctx,
            aggregator_name='fed_avg',
            aggregate_type='weighted_mean',
            sample_num=sample_num)
        return aggregator

    @time_decorator('FedAVG')
    def on_federation(
            self,
            ctx: Context,
            aggregator: PlainTextAggregatorClient,
            fed_args: FedArguments,
            args: TrainingArguments,
            model: Optional[nn.Module] = None,
            optimizer: Optional[Optimizer] = None,
            scheduler: Optional[_LRScheduler] = None,
            dataloader: Optional[Tuple[DataLoader]] = None,
            control: Optional[TrainerControl] = None,
            state: Optional[TrainerState] = None,
            **kwargs):

        aggregator.model_aggregation(model)


class FedAVGServer(FedTrainerServer):

    def __init__(self,
                 ctx: Context,
                 training_args: TrainingArguments = None,
                 fed_args: FedArguments = None,
                 parameter_alignment: bool = True,
                 local_mode: bool = False
                 ) -> None:

        super().__init__(ctx, training_args, fed_args, parameter_alignment, local_mode)

    def init_aggregator(self):
        aggregator = PlainTextAggregatorServer(
            self.ctx, aggregator_name='fed_avg')
        return aggregator

    def on_federation(
            self,
            ctx: Context,
            aggregator: Aggregator,
            fed_args: FedArguments,
            args: TrainingArguments):
        aggregator.model_aggregation()


class FedAVG(object):

    client = FedAVGCLient
    server = FedAVGServer
