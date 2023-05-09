import time
from fate.ml.nn.trainer.trainer_base import FedTrainerClient, FedTrainerServer, logger
from fate.ml.nn.trainer.trainer_base import FedArguments, time_decorator
from dataclasses import field
from dataclasses import dataclass, field
from enum import Enum
from fate.interface import Context
from dataclasses import dataclass
from typing import List, Optional, Tuple, Callable
from fate.interface import Context
from torch.optim import Optimizer
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import _LRScheduler
from transformers.trainer_callback import TrainerCallback
from torch.nn import Module
from torch import nn
from torch.utils.data import DataLoader
from fate.ml.aggregator.plaintext_aggregator import PlainTextAggregatorClient, PlainTextAggregatorServer
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
                 callbacks: List[TrainerCallback] = [], 
                 data_collator: Callable=None,
                 use_hf_default_behavior: bool = False, 
                 compute_metrics: Callable = None, 
                 local_model: bool = False):
        
        super().__init__(ctx, model, loss_fn, optimizer, training_args, fed_args, train_set, val_set, data_collator,
                                  scheduler, callbacks, use_hf_default_behavior,
                                 compute_metrics=compute_metrics, local_mode=local_model)

        self._start_time = 0
        self._start_log = '**********Epoch {}**********'
        self._cur_start_log = None

    def init_aggregator(self):
        sample_num = len(self.train_dataset)
        aggregator = PlainTextAggregatorClient(self.ctx, aggregator_name='fed_avg', aggregate_type='weighted_mean', sample_num=sample_num)
        return aggregator
    
    def on_epoch_begin(
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

        self._cur_start_log = self._start_log.format(int(state.epoch))
        logger.info(self._cur_start_log)
        logger.info('local training start')
        self._start_time = time.time()
    
    @time_decorator('FedAVG')
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
        
        logger.info('epoch {} local training finised, takes {}'.format(state.epoch, time.time() - self._start_time))
        self._start_time = 0
        logger.info('epoch {} model aggregation start'.format(int(state.epoch)))
        aggregator.model_aggregation(model)
        logger.info('epoch {} model aggregation finished'.format(int(state.epoch)))

    def on_log(
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
        
        logger.info('info: {}'.format(state.log_history[-1]))


class FedAVGServer(FedTrainerServer):

    def __init__(self, ctx: Context, training_args: TrainingArguments, fed_args: FedArguments) -> None:
        super().__init__(ctx, training_args, fed_args)

    def init_aggregator(self):
        aggregator = PlainTextAggregatorServer(self.ctx, aggregator_name='fed_avg')
        return aggregator

    def on_epoch_end(self, ctx: Context, aggregator: PlainTextAggregatorServer, fed_args: FedArguments, args: TrainingArguments):
        aggregator.model_aggregation()
