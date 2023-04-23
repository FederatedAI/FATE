from fate.ml.nn.trainer.trainer_base import FedTrainerClient, FedTrainerServer, logger
from fate.ml.nn.trainer.trainer_base import FedArguments
from dataclasses import field
from dataclasses import dataclass, field
from enum import Enum
from fate.interface import Context
from dataclasses import dataclass
from typing import List, Optional, Tuple
from fate.interface import Context
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import _LRScheduler
from transformers.trainer_callback import TrainerCallback
from torch.nn import Module
from torch import nn
from torch.utils.data import DataLoader
from fate.ml.aggregator.plaintext_aggregator import PlainTextAggregatorClient, PlainTextAggregatorServer, Aggregator
from transformers import TrainingArguments, TrainerState, TrainerControl


class AggregateStrategy(Enum):
    EPOCH = "epoch"
    BATCH = "batch"


@dataclass
class FedAVGArguments(FedArguments):

    """
    The arguemnt for FedAVG algorithm, used in FedAVGClient and FedAVGServer.

    Attributes:
        aggregate_strategy: AggregateStrategy
            Aggregate strategy to be used, either 'epoch' or 'batch'.
        aggregate_freq: int
            The frequency of the aggregation, specified as an integer.
        weighted_aggregate: bool
            Whether to use weighted aggregation or not.
        secure_aggregate: bool
            Whether to use secure aggregation or not.
    """
        
    aggregate_strategy: AggregateStrategy = field(default=AggregateStrategy.EPOCH)
    aggregate_freq: int = field(default=1)
    weighted_aggregate: bool = field(default=True)
    secure_aggregate: bool = field(default=False)


class FedAVGCLient(FedTrainerClient):
    
    def __init__(self, ctx: Context, model: Module, loss_fn: Module, optimizer: Optimizer, 
                 training_args: TrainingArguments, fed_args: FedArguments, 
                 train_set: Dataset, val_set: Dataset = None, 
                 scheduler: _LRScheduler = None, 
                 callbacks: List[TrainerCallback] = [], use_hf_default_behavior: bool = False):
        
        super().__init__(ctx, model, loss_fn, optimizer, training_args, fed_args, train_set, val_set, scheduler, callbacks, use_hf_default_behavior)

    
    def init_aggregator(self):
        logger.info('initializing aggregator 2')
        sample_num = len(self.train_dataset)
        aggregator = PlainTextAggregatorClient(self.ctx, aggregator_name='fed_avg', aggregate_type='weighted_mean', sample_num=sample_num)
        return aggregator
    
    def on_epoch_end(
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
        
        logger.info('client on epoch end, aggregating: {}'.format(list(model.parameters())))
        aggregator.model_aggregation(model)
        logger.info('client on epoch end, aggregated done {}'.format(list(model.parameters())))


class FedAVGServer(FedTrainerServer):

    def __init__(self, ctx: Context, training_args: TrainingArguments, fed_args: FedArguments) -> None:
        super().__init__(ctx, training_args, fed_args)

    def init_aggregator(self):
        aggregator = PlainTextAggregatorServer(self.ctx, aggregator_name='fed_avg')
        return aggregator

    def on_epoch_end(self, ctx: Context, aggregator: PlainTextAggregatorServer, fed_args: FedArguments, args: TrainingArguments):

        logger.info('server on epoch end, aggregating')
        aggregator.model_aggregation()
        logger.info('server on epoch end, aggregated done')
