import torch
import math
import sys
from torch import nn
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from enum import Enum
from transformers.training_args import TrainingArguments
from fate.interface import Context
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments as hf_TrainingArguments
from transformers import Trainer, TrainerState, TrainerControl, EvalPrediction
from transformers.trainer_utils import has_length
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR
from torch.utils.data import _utils
from fate.ml.aggregator.base import Aggregator
from transformers.trainer import logger
from transformers.trainer_callback import TrainerCallback, PrinterCallback, TrainerControl, TrainerState
from typing import Optional
import time


def time_decorator(descr=""):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{descr} takes {end_time - start_time:.2f} seconds.")
            return result
        return wrapper
    return decorator


"""
Fed Arguments
"""


class AggregateStrategy(Enum):
    EPOCH = "epoch"
    STEP = "step"
    PROGRESS_PERCENTAGE = "progress_percentage"


@dataclass
class FedArguments(object):
    """
    The argument for Fed algorithm
    """
    aggregate_strategy: AggregateStrategy = field(default=AggregateStrategy.EPOCH.value)
    aggregate_freq: int = field(default=1)
    aggregation_percentage: float = field(default=0.01)

    

@dataclass
class TrainingArguments(hf_TrainingArguments):
    
    # By default, we disable tqdm progress bar for logging conerns.
    output_dir: str = field(default="./")
    disable_tqdm: bool = field(default=True)
    save_strategy: str = field(default="no")
    logging_strategy: str = field(default="epoch")
    evaluation_strategy: str = field(default="no")


"""
Fed Callback Related Classes
"""


class FedCallBackInterFace(object):

    def __init__(self) -> None:
        pass

    def on_init_end(
            self,
            ctx: Context,
            aggregator: Aggregator,
            fed_args: FedArguments,
            args: TrainingArguments,
            model: Optional[nn.Module] = None,
            optimizer: Optional[Optimizer] = None,
            scheduler: Optional[_LRScheduler] = None,
            dataloader: Optional[Tuple[DataLoader]] = None,
            control: Optional[TrainerControl] = None,
            state: Optional[TrainerState] = None,
            **kwargs):
        pass

    def on_train_begin(
            self,
            ctx: Context,
            aggregator: Aggregator,
            fed_args: FedArguments,
            args: TrainingArguments,
            model: Optional[nn.Module] = None,
            optimizer: Optional[Optimizer] = None,
            scheduler: Optional[_LRScheduler] = None,
            dataloader: Optional[Tuple[DataLoader]] = None,
            control: Optional[TrainerControl] = None,
            state: Optional[TrainerState] = None,
            **kwargs):
        pass

    def on_train_end(
            self,
            ctx: Context,
            aggregator: Aggregator,
            fed_args: FedArguments,
            args: TrainingArguments,
            model: Optional[nn.Module] = None,
            optimizer: Optional[Optimizer] = None,
            scheduler: Optional[_LRScheduler] = None,
            dataloader: Optional[Tuple[DataLoader]] = None,
            control: Optional[TrainerControl] = None,
            state: Optional[TrainerState] = None,
            **kwargs):
        pass

    def on_epoch_begin(
            self,
            ctx: Context,
            aggregator: Aggregator,
            fed_args: FedArguments,
            args: TrainingArguments,
            model: Optional[nn.Module] = None,
            optimizer: Optional[Optimizer] = None,
            scheduler: Optional[_LRScheduler] = None,
            dataloader: Optional[Tuple[DataLoader]] = None,
            control: Optional[TrainerControl] = None,
            state: Optional[TrainerState] = None,
            **kwargs):
        pass

    def on_epoch_end(
            self,
            ctx: Context,
            aggregator: Aggregator,
            fed_args: FedArguments,
            args: TrainingArguments,
            model: Optional[nn.Module] = None,
            optimizer: Optional[Optimizer] = None,
            scheduler: Optional[_LRScheduler] = None,
            dataloader: Optional[Tuple[DataLoader]] = None,
            control: Optional[TrainerControl] = None,
            state: Optional[TrainerState] = None,
            **kwargs):
        pass

    def on_step_begin(
            self,
            ctx: Context,
            aggregator: Aggregator,
            fed_args: FedArguments,
            args: TrainingArguments,
            model: Optional[nn.Module] = None,
            optimizer: Optional[Optimizer] = None,
            scheduler: Optional[_LRScheduler] = None,
            dataloader: Optional[Tuple[DataLoader]] = None,
            control: Optional[TrainerControl] = None,
            state: Optional[TrainerState] = None,
            **kwargs):
        pass

    def on_substep_end(
            self,
            ctx: Context,
            aggregator: Aggregator,
            fed_args: FedArguments,
            args: TrainingArguments,
            model: Optional[nn.Module] = None,
            optimizer: Optional[Optimizer] = None,
            scheduler: Optional[_LRScheduler] = None,
            dataloader: Optional[Tuple[DataLoader]] = None,
            control: Optional[TrainerControl] = None,
            state: Optional[TrainerState] = None,
            **kwargs):
            pass

    def on_step_end(
        self,
        ctx: Context,
        aggregator: Aggregator,
        fed_args: FedArguments,
        args: TrainingArguments,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        dataloader: Optional[Tuple[DataLoader]] = None,
        control: Optional[TrainerControl] = None,
        state: Optional[TrainerState] = None,
        **kwargs):
        pass

    def on_log(
        self,
        ctx: Context,
        aggregator: Aggregator,
        fed_args: FedArguments,
        args: TrainingArguments,
        model: Optional[nn.Module] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        dataloader: Optional[Tuple[DataLoader]] = None,
        control: Optional[TrainerControl] = None,
        state: Optional[TrainerState] = None,
        **kwargs):
        pass

    def init_aggregator(self):
        raise NotImplementedError('init_aggregator() must be implemented in subclass, init aggregator here')
    


# I dont like huggingface logging
class LogSuppressFilter(logging.Filter):
    def filter(self, record):
        suppress_list = set(
            ["\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"]
        )
        if record.getMessage() in suppress_list:
            return False
        return True
    

class AggregationChecker:

    def __init__(self, fed_args: FedArguments, max_epoch: int, max_steps: int, epochs_trained: int, steps_trained: int):

        self.fed_args = fed_args
        self.max_epoch = max_epoch
        self.max_steps = max_steps
        self.epochs_trained = epochs_trained
        self.steps_trained = steps_trained
        self.aggregation_count = 0
        self.aggregate_freq = None

        if fed_args.aggregate_strategy == AggregateStrategy.PROGRESS_PERCENTAGE.value and fed_args.aggregation_percentage is not None:
            self.max_aggregation = int(1 / fed_args.aggregation_percentage)
            self.aggregate_freq = int(self.max_steps / self.max_aggregation)
            self.max_aggregation = self.max_aggregation - int(steps_trained / self.aggregate_freq)

        elif fed_args.aggregate_strategy == AggregateStrategy.EPOCH.value:
            self.aggregate_freq = fed_args.aggregate_freq
            self.max_aggregation = int((self.max_epoch - self.epochs_trained) / self.aggregate_freq)

        elif fed_args.aggregate_strategy == AggregateStrategy.STEP.value:
            self.aggregate_freq = fed_args.aggregate_freq
            self.max_aggregation = int((self.max_steps - self.steps_trained) / self.aggregate_freq)

    def should_aggregate(self, state: TrainerState) -> bool:

        cur_epoch = int(state.epoch)
        cur_step = int(state.global_step)
        
        if self.aggregation_count >= self.max_aggregation:
            return False

        if cur_epoch > self.max_epoch:
            return False

        strategy = self.fed_args.aggregate_strategy

        if strategy == AggregateStrategy.EPOCH.value:
            if cur_epoch > self.epochs_trained and (cur_epoch - self.epochs_trained) % self.fed_args.aggregate_freq == 0:
                self.aggregation_count += 1
                return True
        elif strategy == AggregateStrategy.STEP.value:
            if cur_step > self.steps_trained_in_current_epoch and (cur_step - self.steps_trained_in_current_epoch) % self.fed_args.aggregate_freq == 0:
                self.aggregation_count += 1
                return True
        elif strategy == AggregateStrategy.PROGRESS_PERCENTAGE.value:
            if self.aggregate_step_interval is not None:
                total_trained_steps = self.epochs_trained * self.num_update_steps_per_epoch + self.steps_trained_in_current_epoch
                if (cur_epoch * self.num_update_steps_per_epoch + cur_step) % self.aggregate_step_interval == total_trained_steps % self.aggregate_step_interval:
                    self.aggregation_count += 1
                    return True

        return False


class FedParameterAlignCallback(TrainerCallback):

    def __init__(self, ctx, training_args, fed_args, is_server=False) -> None:
        super().__init__()
        self.ctx = ctx
        self.is_server = is_server
        self.training_args = training_args
        self.fed_args = fed_args

    def _client_send_parameters(self, state: TrainerState, args, train_dataloader):
        # client need to compute: epochs, max_steps, num_step_per_epoch, trained_epoch, trained_steps
        # and sync with server

        # compute num_train_epochs, max_steps
        len_dataloader = None

        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)

        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps

        # warm start related variables
        epochs_trained = state.global_step // num_update_steps_per_epoch
        if not args.ignore_data_skip:
            steps_trained_in_current_epoch = state.global_step % (num_update_steps_per_epoch)
            steps_trained_in_current_epoch *= args.gradient_accumulation_steps
        else:
            steps_trained_in_current_epoch = 0
        
        print(num_train_epochs)
        print(num_update_steps_per_epoch)
        print(epochs_trained)
        print(steps_trained_in_current_epoch)

    def _server_check_parameters(self):
        # check if all clients parameters of aggregation match
        print('receiving parameters')

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.is_server:
            self._server_check_parameters()
        else:
            train_dataloader = kwargs['train_dataloader']
            self._client_send_parameters(state, args, train_dataloader)
    

class FedTrainerCallbackWrapper(TrainerCallback):

    def __init__(self, ctx: Context, wrapped_trainer: 'StdFedTrainerMixin'):
        self.ctx = ctx
        self.wrapped_trainer = wrapped_trainer
        self.aggregator = wrapped_trainer.aggregator
        self.fed_arg = wrapped_trainer._fed_args

    def _call_wrapped(self, event_name: str, **kwargs):
        event = getattr(self.wrapped_trainer, event_name)
        kwargs['scheduler'] = kwargs.pop('lr_scheduler', None)

        train_dataloader = kwargs.pop('train_dataloader', None)
        eval_dataloader = kwargs.pop('eval_dataloader', None)
        dataloaders = tuple(filter(None, (train_dataloader, eval_dataloader)))
        kwargs['dataloader'] = dataloaders

        return event(self.ctx, self.aggregator, self.fed_arg, **kwargs)

    def on_init_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs):
        return self._call_wrapped(
            'on_init_end',
            args=args,
            state=state,
            control=control,
            **kwargs)

    def on_train_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs):
        return self._call_wrapped(
            'on_train_begin',
            args=args,
            state=state,
            control=control,
            **kwargs)

    def on_train_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs):
        return self._call_wrapped(
            'on_train_end',
            args=args,
            state=state,
            control=control,
            **kwargs)

    def on_epoch_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs):
        return self._call_wrapped(
            'on_epoch_begin',
            args=args,
            state=state,
            control=control,
            **kwargs)

    def on_epoch_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs):
        return self._call_wrapped(
            'on_epoch_end',
            args=args,
            state=state,
            control=control,
            **kwargs)

    def on_step_begin(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs):
        return self._call_wrapped(
            'on_step_begin',
            args=args,
            state=state,
            control=control,
            **kwargs)

    def on_substep_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs):
        return self._call_wrapped(
            'on_substep_end',
            args=args,
            state=state,
            control=control,
            **kwargs)

    def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs):
        return self._call_wrapped(
            'on_step_end',
            args=args,
            state=state,
            control=control,
            **kwargs)
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return self._call_wrapped(
            'on_log',
            args=args,
            state=state,
            control=control,
            **kwargs)


logger.addFilter(LogSuppressFilter())


"""
Mixin Class For Federation Trainer
"""


class StdFedTrainerMixin(FedCallBackInterFace):

    def __init__(self,
                 ctx: Context,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_args: TrainingArguments,
                 fed_args: FedArguments,
                 train_set: Dataset,
                 val_set: Dataset = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 callbacks: Optional[List[TrainerCallback]] = [],
                 use_hf_default_behavior: bool = False,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 local_mode: bool = None,
                 parameter_alignment = True
                 ):
        
        assert isinstance(
            callbacks, list), 'callback must be a list containing Callback objects, but got {}'.format(
            callbacks)
        
        self.ctx = ctx
        self.local_mode = local_mode
        self.parameter_alignment = parameter_alignment
        self._callbacks = callbacks
        self._args = training_args
        self._fed_args = fed_args
        self._user_compute_metric_func = compute_metrics
        self.train_dataset = train_set
        self.eval_dataset = val_set
        self.loss_func = loss_fn
        self._use_hf_default_behavior = use_hf_default_behavior

        if not self.local_mode:
            self._aggregator: Aggregator = self.init_aggregator()
        else:
            self._aggregator = None
            logger.info('Local model is set, skip initializing aggregator')

    @property
    def aggregator(self):
        if self._aggregator is None:
            raise RuntimeError('Aggregator is not initialized')
        return self._aggregator
        
    def _compute_metrics_warp_func(self, *args, **kwargs):
        
        if self._user_compute_metric_func is None:
            return {}
        else:
            eval_result = self._user_compute_metric_func(*args, **kwargs)
            # Do some FATEBoard Callback here
            return eval_result
    
    def _handle_callback(self, callback_handler, new_callbacks):

        # remove default printer callback, need to use our logging strategy
        new_callback_list = []
        for i in callback_handler.callbacks:
            # if not isinstance(i, PrinterCallback):
            new_callback_list.append(i)
        new_callback_list += new_callbacks
        callback_handler.callbacks = new_callback_list

    def _add_fed_callback(self, callback_handler):
        # the callback handler is Trainer.callback_handler
        if self.local_mode:
            logger.info('Local model is set, federation callback disabled')
            return

        # has_fed_callback = False
        # for c in callback_handler.callbacks:
        #     if isinstance(c, type(fed_callback_inst)):
        #         has_fed_callback = True
        #         break
        # if not has_fed_callback:

        fed_callback_inst = FedTrainerCallbackWrapper(self.ctx, self)
        callback_handler.callbacks.append(fed_callback_inst)
        if self.parameter_alignment:
            callback_handler.callbacks.append(FedParameterAlignCallback(self.ctx, 
                                                                        fed_args=self._fed_args, 
                                                                        training_args=self._args, 
                                                                        is_server=False))
        else:
            logger.warning('Parameter alignment is disabled, this may cause fed-training failure')
        
    def _remove_fed_callback(self, callback_class):
        self.callback_handler.callbacks = [
            c for c in self.callback_handler.callbacks if not isinstance(
                c, callback_class)]


"""
Base Classes of Client/Sever Trainer
"""

class FedTrainerClient(Trainer, StdFedTrainerMixin):

    """
    FedTrainerClient is designed to handle diverse federated training tasks.

    By extending the transformers.Trainer class, this class allows customization of the federated training,
    evaluation, and prediction processes to meet the needs of specific federateion training tasks. Users can
    override relevant methods to implement custom functionality.
    """

    def __init__(self,
                 ctx: Context,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_args: TrainingArguments,
                 fed_args: FedArguments,
                 train_set: Dataset,
                 val_set: Dataset = None,
                 data_collator: Callable = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 callbacks: Optional[List[TrainerCallback]] = [],
                 use_hf_default_behavior: bool = False,
                 compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
                 local_mode: bool = False,
                 parameter_alignment = True
                 ):

        # default use no lr decay
        if scheduler is None:
            if use_hf_default_behavior and optimizer is None:
                pass
            else:
                scheduler = LambdaLR(optimizer, lambda x: 1)
    
        # in case you forget to set evaluation_strategy
        if val_set is not None and training_args.evaluation_strategy == 'no':
            training_args.evaluation_strategy = 'epoch'

        
        StdFedTrainerMixin.__init__(self,
                                    ctx=ctx,
                                    model=model,
                                    loss_fn=loss_fn,
                                    optimizer=optimizer,
                                    training_args=training_args,
                                    fed_args=fed_args,
                                    train_set=train_set,
                                    val_set=val_set,
                                    scheduler=scheduler,
                                    callbacks=callbacks,
                                    use_hf_default_behavior=use_hf_default_behavior,
                                    compute_metrics=compute_metrics,
                                    local_mode=local_mode,
                                    parameter_alignment=parameter_alignment
                                    )

        if data_collator is None:
            data_collator = _utils.collate.default_collate

        Trainer.__init__(self,
                        model=model,
                        args=self._args,
                        train_dataset=train_set,
                        eval_dataset=val_set,
                        data_collator=data_collator,
                        optimizers=[optimizer, scheduler],
                        compute_metrics=self._compute_metrics_warp_func
                        )
        
        # self._handle_callback(self.callback_handler, self._callbacks)
        self._add_fed_callback(self.callback_handler)

    def init_aggregator(self):
        return None

    def compute_loss(self, model, inputs, **kwargs):

        if self._use_hf_default_behavior:
            return super().compute_loss(model, inputs, **kwargs)
        else:
            feats, labels = inputs
            logits = model(feats)
            loss = self.loss_func(logits.flatten(), labels.flatten())
            return loss

    def prediction_step(self,
                        model: nn.Module,
                        inputs: Dict[str,
                                     Union[torch.Tensor,
                                           Any]],
                        prediction_loss_only: bool,
                        ignore_keys: Optional[List[str]] = None):
        
        if self._use_hf_default_behavior:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        else:
            with torch.no_grad():
                feats, labels = inputs
                logits = model(feats)
            return (None, logits, labels)
        

class FedTrainerServer(FedCallBackInterFace):

    def __init__(self, ctx: Context, 
                 parameter_alignment: bool = True,
                 training_args: TrainingArguments = None, 
                 fed_args: FedArguments = None) -> None:
        
        self.ctx = ctx
        self.parameter_alignment = parameter_alignment
        self.aggregator: Aggregator = self.init_aggregator()
        self._args = training_args
        self._fed_args = fed_args
        self._max_steps = None
        self._parameter_check_callback = FedParameterAlignCallback(self.ctx, None, None, True)

    def init_aggregator(self):
        return None

    def train(self):

        if self.parameter_alignment:
            pass
        else:
            epochs = self._args.num_train_epochs
            max_steps = self._args.max_steps
        
        self.on_init_end(self.ctx, aggregator=self.aggregator, args=self._args, fed_args=self._fed_args)
        self.on_train_begin(self.ctx, aggregator=self.aggregator, args=self._args, fed_args=self._fed_args)
        for epoch in range(self._args.num_train_epochs):
            self.on_epoch_begin(self.ctx, aggregator=self.aggregator, args=self._args, fed_args=self._fed_args)
            # for step in range(self.training_args.num_steps_per_epoch):
            #     self.on_step_begin(self.ctx, aggregator=self.aggregator, args=self._args, fed_args=self._fed_args, step=step)
            #     self.on_substep_end(self.ctx, aggregator=self.aggregator, args=self._args, fed_args=self._fed_args)
            #     self.on_step_end(self.ctx, aggregator=self.aggregator, args=self._args, fed_args=self._fed_args, step=step)
            self.on_epoch_end(self.ctx, aggregator=self.aggregator, args=self._args, fed_args=self._fed_args)
        self.on_train_end(self.ctx, aggregator=self.aggregator, args=self._args, fed_args=self._fed_args)

