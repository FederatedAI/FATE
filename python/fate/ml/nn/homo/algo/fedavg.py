import time

from transformers.training_args import TrainingArguments
from fate.ml.aggregator.base import Aggregator
from fate.ml.nn.trainer.trainer_base import FedTrainerClient, FedTrainerServer, TrainingArguments, logger
from fate.ml.nn.trainer.trainer_base import FedArguments, time_decorator
from dataclasses import field
from dataclasses import dataclass, field
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
from fate.ml.nn.utils.algo import HomoAlgorithm



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
    
    def __init__(self, model: Module, loss_fn: Module, optimizer: Optimizer, 
                 training_args: TrainingArguments, fed_args: FedArguments, 
                 train_set: Dataset, val_set: Dataset = None, 
                 scheduler: _LRScheduler = None, 
                 callbacks: List[TrainerCallback] = [], 
                 data_collator: Callable=None,
                 use_hf_default_behavior: bool = False, 
                 compute_metrics: Callable = None, 
                 local_model: bool = False,
                 ctx: Context = None
                 ):
        
        super().__init__(model, loss_fn, optimizer, training_args, fed_args, train_set, val_set, data_collator,
                         scheduler, callbacks, use_hf_default_behavior,
                         compute_metrics=compute_metrics, local_mode=local_model)

    def init_aggregator(self):
        sample_num = len(self.train_dataset)
        aggregator = PlainTextAggregatorClient(self.ctx, aggregator_name='fed_avg', aggregate_type='weighted_mean', sample_num=sample_num)
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
                 parameter_alignment: bool = True,
                 training_args: TrainingArguments = None, 
                 fed_args: FedArguments = None,
                 ctx: Context = None, 
                 ) -> None:
        
        super().__init__(ctx, parameter_alignment, training_args, fed_args)

    def init_aggregator(self):
        aggregator = PlainTextAggregatorServer(self.ctx, aggregator_name='fed_avg')
        return aggregator

    def on_federation(self, ctx: Context, aggregator: Aggregator, fed_args: FedArguments, args: TrainingArguments):
        aggregator.model_aggregation()


class FedAVG(HomoAlgorithm):

    client = FedAVGCLient
    sever = FedAVGServer
